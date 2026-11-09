// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Utils/ConstantEvaluator.h"

#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstring>
#include <optional>

namespace mlir::syna::torq {

namespace {

// Loops with more iterations than this go to the JIT.
constexpr int64_t kMaxLoopIterations = 1 << 16;

// Computes compile-time constants directly from the data of the constants they depend on, so the
// common cones (weight layouts, slices, integer generics, loops with constant bounds)
// don't need the JIT. It gives up on anything it doesn't support and on results that contain
// elements no op has written; the JIT computes those.
class Evaluator {
  public:
    Evaluator(bool recursive, ArrayRef<Value> assumeZero)
        : recursive(recursive), assumeZero(assumeZero) {}

    // Returns the value of `value`, or null if it can't be computed here.
    DenseElementsAttr evaluate(Value value) {
        auto type = dyn_cast<RankedTensorType>(value.getType());
        if (!type || !type.hasStaticShape()) {
            return nullptr;
        }
        rootOp = value.getDefiningOp();
        auto result = get(value);
        if (failed(result) || !result->tensor) {
            return nullptr;
        }
        const Tensor &tensor = *result->tensor;
        if (!llvm::all_of(tensor.defined, [](bool defined) { return defined; })) {
            return nullptr;
        }
        return DenseElementsAttr::getFromRawBuffer(type, tensor.data);
    }

  private:
    // The elements of a tensor in row-major order, and which of them were written.
    struct Tensor {
        RankedTensorType type;
        size_t elementSize = 0;
        std::vector<char> data;
        std::vector<bool> defined;
    };

    // The value of an SSA value: a scalar attribute or a tensor.
    struct Eval {
        Attribute scalar;
        std::shared_ptr<Tensor> tensor;
    };

    static std::shared_ptr<Tensor> newTensor(Type type) {
        auto tensorType = dyn_cast<RankedTensorType>(type);
        if (!tensorType || !tensorType.hasStaticShape() ||
            !tensorType.getElementType().isIntOrFloat()) {
            return nullptr;
        }
        unsigned bitWidth = tensorType.getElementType().getIntOrFloatBitWidth();
        if (bitWidth % 8 != 0 || bitWidth > 64) {
            return nullptr;
        }
        auto tensor = std::make_shared<Tensor>();
        tensor->type = tensorType;
        tensor->elementSize = bitWidth / 8;
        tensor->data.resize(tensorType.getNumElements() * tensor->elementSize);
        tensor->defined.resize(tensorType.getNumElements(), false);
        return tensor;
    }

    static Attribute readElement(const Tensor &tensor, int64_t pos) {
        Type elementType = tensor.type.getElementType();
        uint64_t bits = 0;
        std::memcpy(&bits, &tensor.data[pos * tensor.elementSize], tensor.elementSize);
        APInt value(elementType.getIntOrFloatBitWidth(), bits);
        if (auto floatType = dyn_cast<FloatType>(elementType)) {
            return FloatAttr::get(floatType, APFloat(floatType.getFloatSemantics(), value));
        }
        return IntegerAttr::get(elementType, value);
    }

    static bool writeElement(Tensor &tensor, int64_t pos, Attribute attr) {
        APInt value;
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
            value = intAttr.getValue();
        }
        else if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
            value = floatAttr.getValue().bitcastToAPInt();
        }
        else {
            return false;
        }
        if (value.getBitWidth() != tensor.elementSize * 8) {
            return false;
        }
        uint64_t bits = value.getZExtValue();
        std::memcpy(&tensor.data[pos * tensor.elementSize], &bits, tensor.elementSize);
        tensor.defined[pos] = true;
        return true;
    }

    static void copyElement(const Tensor &from, int64_t fromPos, Tensor &to, int64_t toPos) {
        std::memcpy(
            &to.data[toPos * to.elementSize], &from.data[fromPos * from.elementSize], to.elementSize
        );
        to.defined[toPos] = from.defined[fromPos];
    }

    static SmallVector<int64_t> stridesOf(ArrayRef<int64_t> shape) {
        SmallVector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    // Calls `fn` for each index of `shape` in row-major order, with the index and its position.
    static void
    forEachIndex(ArrayRef<int64_t> shape, llvm::function_ref<void(ArrayRef<int64_t>, int64_t)> fn) {
        int64_t count = 1;
        for (int64_t size : shape) {
            count *= size;
        }
        SmallVector<int64_t> index(shape.size(), 0);
        for (int64_t pos = 0; pos < count; ++pos) {
            fn(index, pos);
            for (int64_t i = static_cast<int64_t>(shape.size()) - 1;
                 i >= 0 && ++index[i] == shape[i]; --i) {
                index[i] = 0;
            }
        }
    }

    FailureOr<int64_t> getInt(OpFoldResult ofr) {
        Attribute attr = dyn_cast<Attribute>(ofr);
        if (!attr) {
            auto eval = get(cast<Value>(ofr));
            if (failed(eval)) {
                return failure();
            }
            attr = eval->scalar;
        }
        auto intAttr = dyn_cast_or_null<IntegerAttr>(attr);
        if (!intAttr) {
            return failure();
        }
        return intAttr.getInt();
    }

    FailureOr<SmallVector<int64_t>> getInts(ArrayRef<OpFoldResult> ofrs) {
        SmallVector<int64_t> ints;
        for (auto ofr : ofrs) {
            auto value = getInt(ofr);
            if (failed(value)) {
                return failure();
            }
            ints.push_back(*value);
        }
        return ints;
    }

    std::shared_ptr<Tensor> getTensor(Value value) {
        auto eval = get(value);
        return succeeded(eval) ? eval->tensor : nullptr;
    }

    FailureOr<Eval> get(Value value) {
        if (auto it = env.find(value); it != env.end()) {
            return it->second;
        }
        if (llvm::is_contained(assumeZero, value)) {
            return failure();
        }

        Attribute attr;
        if (matchPattern(value, m_Constant(&attr))) {
            Eval eval;
            if (isa<IntegerAttr, FloatAttr>(attr)) {
                eval.scalar = attr;
            }
            else if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
                eval.tensor = newTensor(value.getType());
                if (!eval.tensor) {
                    return failure();
                }
                ArrayRef<char> raw = dense.getRawData();
                size_t elementSize = eval.tensor->elementSize;
                if (dense.isSplat()) {
                    for (size_t pos = 0; pos < eval.tensor->defined.size(); ++pos) {
                        std::memcpy(&eval.tensor->data[pos * elementSize], raw.data(), elementSize);
                    }
                }
                else if (raw.size() == eval.tensor->data.size()) {
                    std::memcpy(eval.tensor->data.data(), raw.data(), raw.size());
                }
                else {
                    return failure();
                }
                eval.tensor->defined.assign(eval.tensor->defined.size(), true);
            }
            else {
                return failure();
            }
            env[value] = eval;
            return eval;
        }

        // Block arguments of the regions evaluated here are always in env.
        Operation *op = value.getDefiningOp();
        if (!op || !rootOp) {
            return failure();
        }
        // Without `recursive`, only the root op (and what its regions contain) is computed: the
        // values it uses must be constants.
        if (!recursive && op != rootOp && !rootOp->isProperAncestor(op)) {
            return failure();
        }
        if (failed(evalOp(op))) {
            return failure();
        }
        if (auto it = env.find(value); it != env.end()) {
            return it->second;
        }
        return failure();
    }

    LogicalResult setTensorResult(Operation *op, std::shared_ptr<Tensor> tensor) {
        if (!tensor) {
            return failure();
        }
        env[op->getResult(0)] = Eval{nullptr, std::move(tensor)};
        return success();
    }

    std::shared_ptr<Tensor> extractSlice(
        const Tensor &source, ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes,
        ArrayRef<int64_t> strides, Type resultType
    ) {
        auto result = newTensor(resultType);
        if (!result || result->elementSize != source.elementSize) {
            return nullptr;
        }
        auto sourceStrides = stridesOf(source.type.getShape());
        forEachIndex(sizes, [&](ArrayRef<int64_t> index, int64_t pos) {
            int64_t sourcePos = 0;
            for (size_t i = 0; i < index.size(); ++i) {
                sourcePos += (offsets[i] + index[i] * strides[i]) * sourceStrides[i];
            }
            copyElement(source, sourcePos, *result, pos);
        });
        return result;
    }

    // Writes `source` into `dest` (a copy, unless `inPlace`), as tensor.insert_slice does.
    std::shared_ptr<Tensor> insertSlice(
        const Tensor &source, std::shared_ptr<Tensor> dest, ArrayRef<int64_t> offsets,
        ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, bool inPlace = false
    ) {
        if (source.elementSize != dest->elementSize) {
            return nullptr;
        }
        auto result = inPlace ? dest : std::make_shared<Tensor>(*dest);
        auto destStrides = stridesOf(dest->type.getShape());
        forEachIndex(sizes, [&](ArrayRef<int64_t> index, int64_t pos) {
            int64_t destPos = 0;
            for (size_t i = 0; i < index.size(); ++i) {
                destPos += (offsets[i] + index[i] * strides[i]) * destStrides[i];
            }
            copyElement(source, pos, *result, destPos);
        });
        return result;
    }

    // Output dim i is input dim perm[i], as in torq_hl.transpose and linalg.transpose.
    std::shared_ptr<Tensor> transpose(const Tensor &input, ArrayRef<int64_t> perm, Type type) {
        auto result = newTensor(type);
        if (!result || result->elementSize != input.elementSize ||
            perm.size() != static_cast<size_t>(result->type.getRank())) {
            return nullptr;
        }
        auto inputStrides = stridesOf(input.type.getShape());
        forEachIndex(result->type.getShape(), [&](ArrayRef<int64_t> index, int64_t pos) {
            int64_t inputPos = 0;
            for (size_t i = 0; i < index.size(); ++i) {
                inputPos += index[i] * inputStrides[perm[i]];
            }
            copyElement(input, inputPos, *result, pos);
        });
        return result;
    }

    // Same data, other shape (expand_shape, collapse_shape).
    std::shared_ptr<Tensor> reshape(const Tensor &input, Type type) {
        auto result = newTensor(type);
        if (!result || result->data.size() != input.data.size()) {
            return nullptr;
        }
        result->data = input.data;
        result->defined = input.defined;
        return result;
    }

    // Folds `op` with constant scalar operands.
    LogicalResult foldScalarOp(Operation *op, ArrayRef<Attribute> operands) {
        if (op->getNumResults() != 1 || op->getNumRegions() != 0 ||
            isa<ShapedType>(op->getResult(0).getType())) {
            return failure();
        }
        SmallVector<OpFoldResult> foldResults;
        if (failed(op->fold(operands, foldResults)) || foldResults.size() != 1) {
            return failure();
        }
        Attribute attr = dyn_cast<Attribute>(foldResults[0]);
        if (!attr) {
            auto eval = get(cast<Value>(foldResults[0]));
            if (failed(eval)) {
                return failure();
            }
            attr = eval->scalar;
        }
        if (!isa_and_nonnull<IntegerAttr, FloatAttr>(attr)) {
            return failure();
        }
        env[op->getResult(0)] = Eval{attr, nullptr};
        return success();
    }

    // The body of a generic as a list of integer operations on value slots, so it can be run on
    // many elements without creating attributes. Slots: the block arguments, then the values
    // from outside the body, then the results of the body ops.
    struct IntProgram {
        enum class Kind {
            ExtSI,
            ExtUI,
            TruncI,
            AddI,
            SubI,
            MulI,
            MaxSI,
            MinSI,
            MaxUI,
            MinUI,
            ShLI,
            ShRSI,
            ShRUI,
            AndI,
            OrI,
            XOrI,
            CmpI,
            Select
        };
        struct Step {
            Kind kind;
            arith::CmpIPredicate predicate;
            unsigned result;
            SmallVector<unsigned, 3> operands;
        };
        SmallVector<APInt> slots;
        SmallVector<Step> steps;
        unsigned yielded = 0;
    };

    // Builds the program of `body`, or returns nullopt if an op or a type isn't supported.
    std::optional<IntProgram> buildIntProgram(Block &body) {
        IntProgram program;
        DenseMap<Value, unsigned> slotOf;
        auto addSlot = [&](Value value) -> std::optional<unsigned> {
            auto intType = dyn_cast<IntegerType>(value.getType());
            if (!intType || intType.getWidth() > 64) {
                return std::nullopt;
            }
            slotOf[value] = program.slots.size();
            program.slots.push_back(APInt(intType.getWidth(), 0));
            return slotOf[value];
        };
        for (BlockArgument arg : body.getArguments()) {
            if (!addSlot(arg)) {
                return std::nullopt;
            }
        }
        for (Operation &op : body.without_terminator()) {
            using Kind = IntProgram::Kind;
            auto kind = TypeSwitch<Operation *, std::optional<Kind>>(&op)
                            .Case([](arith::ExtSIOp) { return Kind::ExtSI; })
                            .Case([](arith::ExtUIOp) { return Kind::ExtUI; })
                            .Case([](arith::TruncIOp) { return Kind::TruncI; })
                            .Case([](arith::AddIOp) { return Kind::AddI; })
                            .Case([](arith::SubIOp) { return Kind::SubI; })
                            .Case([](arith::MulIOp) { return Kind::MulI; })
                            .Case([](arith::MaxSIOp) { return Kind::MaxSI; })
                            .Case([](arith::MinSIOp) { return Kind::MinSI; })
                            .Case([](arith::MaxUIOp) { return Kind::MaxUI; })
                            .Case([](arith::MinUIOp) { return Kind::MinUI; })
                            .Case([](arith::ShLIOp) { return Kind::ShLI; })
                            .Case([](arith::ShRSIOp) { return Kind::ShRSI; })
                            .Case([](arith::ShRUIOp) { return Kind::ShRUI; })
                            .Case([](arith::AndIOp) { return Kind::AndI; })
                            .Case([](arith::OrIOp) { return Kind::OrI; })
                            .Case([](arith::XOrIOp) { return Kind::XOrI; })
                            .Case([](arith::CmpIOp) { return Kind::CmpI; })
                            .Case([](arith::SelectOp) { return Kind::Select; })
                            .Default([](Operation *) { return std::nullopt; });
            if (!kind) {
                return std::nullopt;
            }
            IntProgram::Step step{*kind, arith::CmpIPredicate::eq, 0, {}};
            if (auto cmpOp = dyn_cast<arith::CmpIOp>(op)) {
                step.predicate = cmpOp.getPredicate();
            }
            for (Value operand : op.getOperands()) {
                if (!slotOf.count(operand)) {
                    if (operand.getParentBlock() == &body) {
                        return std::nullopt;
                    }
                    auto eval = get(operand);
                    auto intAttr = succeeded(eval) ? dyn_cast_or_null<IntegerAttr>(eval->scalar)
                                                   : IntegerAttr();
                    auto slot = addSlot(operand);
                    if (!intAttr || !slot) {
                        return std::nullopt;
                    }
                    program.slots[*slot] = intAttr.getValue();
                }
                step.operands.push_back(slotOf[operand]);
            }
            auto result = addSlot(op.getResult(0));
            if (!result) {
                return std::nullopt;
            }
            step.result = *result;
            program.steps.push_back(step);
        }
        Value yielded = body.getTerminator()->getOperand(0);
        if (!slotOf.count(yielded)) {
            return std::nullopt;
        }
        program.yielded = slotOf[yielded];
        return program;
    }

    // Runs the program on its current slots. Fails on shifts by the bit width or more, which
    // give an undefined result.
    static bool runIntProgram(IntProgram &program) {
        using Kind = IntProgram::Kind;
        auto &slots = program.slots;
        for (auto &step : program.steps) {
            const APInt &a = slots[step.operands[0]];
            const APInt &b = step.operands.size() > 1 ? slots[step.operands[1]] : a;
            APInt &result = slots[step.result];
            unsigned width = result.getBitWidth();
            switch (step.kind) {
            case Kind::ExtSI:
                result = a.sext(width);
                break;
            case Kind::ExtUI:
                result = a.zext(width);
                break;
            case Kind::TruncI:
                result = a.trunc(width);
                break;
            case Kind::AddI:
                result = a + b;
                break;
            case Kind::SubI:
                result = a - b;
                break;
            case Kind::MulI:
                result = a * b;
                break;
            case Kind::MaxSI:
                result = llvm::APIntOps::smax(a, b);
                break;
            case Kind::MinSI:
                result = llvm::APIntOps::smin(a, b);
                break;
            case Kind::MaxUI:
                result = llvm::APIntOps::umax(a, b);
                break;
            case Kind::MinUI:
                result = llvm::APIntOps::umin(a, b);
                break;
            case Kind::ShLI:
            case Kind::ShRSI:
            case Kind::ShRUI:
                if (b.uge(width)) {
                    return false;
                }
                result = step.kind == Kind::ShLI    ? a.shl(b)
                         : step.kind == Kind::ShRSI ? a.ashr(b)
                                                    : a.lshr(b);
                break;
            case Kind::AndI:
                result = a & b;
                break;
            case Kind::OrI:
                result = a | b;
                break;
            case Kind::XOrI:
                result = a ^ b;
                break;
            case Kind::CmpI:
                result = APInt(1, arith::applyCmpPredicate(step.predicate, a, b));
                break;
            case Kind::Select:
                result = a.getBoolValue() ? b : slots[step.operands[2]];
                break;
            }
        }
        return true;
    }

    // Evaluates a linalg.generic or linalg.reduce.
    LogicalResult evalLinalgOp(linalg::LinalgOp linalgOp) {
        if (linalgOp->getNumResults() != 1 || linalgOp.getNumDpsInits() != 1) {
            return failure();
        }
        Block &body = *linalgOp.getBlock();
        bool readsInit = !body.getArguments().back().use_empty();
        bool isReduction = linalgOp.getNumReductionLoops() != 0;
        auto maps = linalgOp.getIndexingMapsArray();
        AffineMap outputMap = maps.back();
        if (isReduction ? !outputMap.isProjectedPermutation() : !outputMap.isPermutation()) {
            return failure();
        }
        auto loopRanges = linalgOp.getStaticLoopRanges();
        if (llvm::any_of(loopRanges, ShapedType::isDynamic)) {
            return failure();
        }

        auto result = newTensor(linalgOp->getResult(0).getType());
        if (!result) {
            return failure();
        }
        // A reduction accumulates into the init: only integer reductions, whose result doesn't
        // depend on the order of the iterations, are computed here.
        if (readsInit) {
            if (isReduction && !result->type.getElementType().isInteger()) {
                return failure();
            }
            auto init = getTensor(linalgOp.getDpsInits()[0]);
            if (!init || init->elementSize != result->elementSize) {
                return failure();
            }
            result->data = init->data;
            result->defined = init->defined;
        }
        else if (isReduction) {
            return failure();
        }

        // Input positions: each map result is a loop dim or a constant index.
        SmallVector<std::shared_ptr<Tensor>> inputs;
        SmallVector<SmallVector<std::pair<int64_t, int64_t>>> inputTerms; // (dim or -1, value)
        for (auto [input, map] : llvm::zip(linalgOp.getDpsInputs(), maps)) {
            auto tensor = getTensor(input);
            if (!tensor) {
                return failure();
            }
            auto strides = stridesOf(tensor->type.getShape());
            SmallVector<std::pair<int64_t, int64_t>> terms;
            for (auto [i, expr] : llvm::enumerate(map.getResults())) {
                if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
                    terms.push_back({dim.getPosition(), strides[i]});
                }
                else if (auto cst = dyn_cast<AffineConstantExpr>(expr)) {
                    terms.push_back({-1, cst.getValue() * strides[i]});
                }
                else {
                    return failure();
                }
            }
            inputs.push_back(tensor);
            inputTerms.push_back(std::move(terms));
        }
        auto resultStrides = stridesOf(result->type.getShape());
        SmallVector<int64_t> outputStrides(loopRanges.size(), 0);
        for (auto [i, expr] : llvm::enumerate(outputMap.getResults())) {
            outputStrides[cast<AffineDimExpr>(expr).getPosition()] = resultStrides[i];
        }
        auto positionOf = [](ArrayRef<std::pair<int64_t, int64_t>> terms, ArrayRef<int64_t> index) {
            int64_t pos = 0;
            for (auto [dim, value] : terms) {
                pos += dim < 0 ? value : index[dim] * value;
            }
            return pos;
        };
        auto outputPosOf = [&](ArrayRef<int64_t> index) {
            int64_t pos = 0;
            for (auto [i, value] : llvm::enumerate(index)) {
                pos += value * outputStrides[i];
            }
            return pos;
        };

        auto yieldOp = cast<linalg::YieldOp>(body.getTerminator());

        // A body that only yields an input is a copy: no folding needed.
        if (auto arg = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
            !isReduction && arg && arg.getOwner() == &body && arg.getArgNumber() < inputs.size()) {
            const Tensor &input = *inputs[arg.getArgNumber()];
            if (input.elementSize != result->elementSize) {
                return failure();
            }
            const auto &terms = inputTerms[arg.getArgNumber()];
            forEachIndex(loopRanges, [&](ArrayRef<int64_t> index, int64_t) {
                copyElement(input, positionOf(terms, index), *result, outputPosOf(index));
            });
            return setTensorResult(linalgOp, result);
        }

        bool ok = true;
        unsigned initArg = body.getNumArguments() - 1;

        // Integer bodies run on APInt values; other bodies go to the JIT.
        auto program = buildIntProgram(body);
        if (!program) {
            return failure();
        }
        forEachIndex(loopRanges, [&](ArrayRef<int64_t> index, int64_t) {
            if (!ok) {
                return;
            }
            for (auto [i, input] : llvm::enumerate(inputs)) {
                int64_t pos = positionOf(inputTerms[i], index);
                if (!input->defined[pos]) {
                    ok = false;
                    return;
                }
                uint64_t bits = 0;
                std::memcpy(&bits, &input->data[pos * input->elementSize], input->elementSize);
                program->slots[i] = APInt(program->slots[i].getBitWidth(), bits);
            }
            int64_t outPos = outputPosOf(index);
            if (readsInit) {
                if (!result->defined[outPos]) {
                    ok = false;
                    return;
                }
                uint64_t bits = 0;
                std::memcpy(
                    &bits, &result->data[outPos * result->elementSize], result->elementSize
                );
                program->slots[initArg] = APInt(program->slots[initArg].getBitWidth(), bits);
            }
            if (!runIntProgram(*program)) {
                ok = false;
                return;
            }
            const APInt &value = program->slots[program->yielded];
            if (value.getBitWidth() != result->elementSize * 8) {
                ok = false;
                return;
            }
            uint64_t bits = value.getZExtValue();
            std::memcpy(&result->data[outPos * result->elementSize], &bits, result->elementSize);
            result->defined[outPos] = true;
        });
        if (!ok) {
            return failure();
        }
        return setTensorResult(linalgOp, result);
    }

    // Evaluates the ops of `block` in order, without its terminator.
    LogicalResult evalBlock(Block &block) {
        for (Operation &op : block.without_terminator()) {
            // Constants are read when they are used.
            if (op.hasTrait<OpTrait::ConstantLike>()) {
                continue;
            }
            if (failed(evalOp(&op))) {
                return failure();
            }
        }
        return success();
    }

    // Removes the values defined in the regions of `loopOp` from env: they hold the last
    // iteration's values, which a later evaluate() of one of them must not return.
    void forgetInnerValues(Operation *loopOp) {
        loopOp->walk([&](Operation *op) {
            for (Region &region : op->getRegions()) {
                for (Block &block : region) {
                    for (BlockArgument arg : block.getArguments()) {
                        env.erase(arg);
                    }
                }
            }
            if (op != loopOp) {
                for (Value result : op->getResults()) {
                    env.erase(result);
                }
            }
        });
    }

    LogicalResult evalFor(scf::ForOp forOp) {
        auto lb = getInt(forOp.getLowerBound());
        auto ub = getInt(forOp.getUpperBound());
        auto step = getInt(forOp.getStep());
        if (failed(lb) || failed(ub) || failed(step) || *step <= 0 ||
            (*ub - *lb) / *step > kMaxLoopIterations) {
            return failure();
        }
        SmallVector<Eval> iterValues;
        for (Value init : forOp.getInitArgs()) {
            auto eval = get(init);
            if (failed(eval)) {
                return failure();
            }
            iterValues.push_back(*eval);
        }
        Block &body = *forOp.getBody();
        auto indexType = forOp.getInductionVar().getType();
        for (int64_t iv = *lb; iv < *ub; iv += *step) {
            env[forOp.getInductionVar()] = Eval{IntegerAttr::get(indexType, iv), nullptr};
            for (auto [arg, value] : llvm::zip(forOp.getRegionIterArgs(), iterValues)) {
                env[arg] = value;
            }
            if (failed(evalBlock(body))) {
                return failure();
            }
            auto yieldOp = cast<scf::YieldOp>(body.getTerminator());
            for (auto [i, operand] : llvm::enumerate(yieldOp.getOperands())) {
                auto eval = get(operand);
                if (failed(eval)) {
                    return failure();
                }
                iterValues[i] = *eval;
            }
        }
        for (auto [result, value] : llvm::zip(forOp.getResults(), iterValues)) {
            env[result] = value;
        }
        return success();
    }

    LogicalResult evalForall(scf::ForallOp forallOp) {
        auto lbs = getInts(forallOp.getMixedLowerBound());
        auto ubs = getInts(forallOp.getMixedUpperBound());
        auto steps = getInts(forallOp.getMixedStep());
        if (failed(lbs) || failed(ubs) || failed(steps)) {
            return failure();
        }
        SmallVector<int64_t> counts;
        int64_t total = 1;
        for (auto [lb, ub, step] : llvm::zip(*lbs, *ubs, *steps)) {
            if (step <= 0) {
                return failure();
            }
            counts.push_back(ub > lb ? (ub - lb + step - 1) / step : 0);
            total *= counts.back();
        }
        if (total > kMaxLoopIterations) {
            return failure();
        }

        // The shared outputs are updated in place: the iterations write disjoint slices.
        SmallVector<std::shared_ptr<Tensor>> shared;
        for (Value output : forallOp.getOutputs()) {
            auto tensor = getTensor(output);
            if (!tensor) {
                return failure();
            }
            shared.push_back(std::make_shared<Tensor>(*tensor));
        }

        Block &body = *forallOp.getBody();
        auto ivs = forallOp.getInductionVars();
        bool ok = true;
        forEachIndex(counts, [&](ArrayRef<int64_t> index, int64_t) {
            if (!ok) {
                return;
            }
            for (auto [i, iv] : llvm::enumerate(ivs)) {
                env[iv] = Eval{
                    IntegerAttr::get(iv.getType(), (*lbs)[i] + index[i] * (*steps)[i]), nullptr
                };
            }
            for (auto [arg, tensor] : llvm::zip(forallOp.getRegionIterArgs(), shared)) {
                env[arg] = Eval{nullptr, tensor};
            }
            if (failed(evalBlock(body))) {
                ok = false;
                return;
            }
            for (Operation &yieldingOp : forallOp.getTerminator().getYieldingOps()) {
                auto insertOp = dyn_cast<tensor::ParallelInsertSliceOp>(&yieldingOp);
                auto dest = insertOp ? dyn_cast<BlockArgument>(insertOp.getDest()) : nullptr;
                auto source = insertOp ? getTensor(insertOp.getSource()) : nullptr;
                auto offsets = insertOp ? getInts(insertOp.getMixedOffsets())
                                        : FailureOr<SmallVector<int64_t>>(failure());
                auto sizes = insertOp ? getInts(insertOp.getMixedSizes())
                                      : FailureOr<SmallVector<int64_t>>(failure());
                auto strides = insertOp ? getInts(insertOp.getMixedStrides())
                                        : FailureOr<SmallVector<int64_t>>(failure());
                if (!dest || !source || failed(offsets) || failed(sizes) || failed(strides)) {
                    ok = false;
                    return;
                }
                auto &target = shared
                    [dest.getArgNumber() - forallOp.getRegionIterArgs().front().getArgNumber()];
                if (!insertSlice(*source, target, *offsets, *sizes, *strides, /*inPlace=*/true)) {
                    ok = false;
                    return;
                }
            }
        });
        if (!ok) {
            return failure();
        }
        for (auto [result, tensor] : llvm::zip(forallOp.getResults(), shared)) {
            env[result] = Eval{nullptr, tensor};
        }
        return success();
    }

    LogicalResult evalOp(Operation *op) {
        return TypeSwitch<Operation *, LogicalResult>(op)
            .Case([&](tensor::EmptyOp emptyOp) {
                return setTensorResult(op, newTensor(emptyOp.getType()));
            })
            .Case([&](linalg::FillOp fillOp) {
                if (fillOp->getNumResults() != 1) {
                    return failure();
                }
                auto value = get(fillOp.getInputs()[0]);
                auto result = newTensor(fillOp->getResult(0).getType());
                if (failed(value) || !value->scalar || !result) {
                    return failure();
                }
                for (size_t pos = 0; pos < result->defined.size(); ++pos) {
                    if (!writeElement(*result, pos, value->scalar)) {
                        return failure();
                    }
                }
                return setTensorResult(op, result);
            })
            .Case([&](tensor::ExtractOp extractOp) -> LogicalResult {
                auto tensor = getTensor(extractOp.getTensor());
                auto indices = getInts(getAsOpFoldResult(extractOp.getIndices()));
                if (!tensor || failed(indices)) {
                    return failure();
                }
                auto strides = stridesOf(tensor->type.getShape());
                int64_t pos = 0;
                for (auto [index, stride] : llvm::zip(*indices, strides)) {
                    pos += index * stride;
                }
                if (!tensor->defined[pos]) {
                    return failure();
                }
                env[op->getResult(0)] = Eval{readElement(*tensor, pos), nullptr};
                return success();
            })
            .Case([&](tensor::ExtractSliceOp sliceOp) {
                auto source = getTensor(sliceOp.getSource());
                auto offsets = getInts(sliceOp.getMixedOffsets());
                auto sizes = getInts(sliceOp.getMixedSizes());
                auto strides = getInts(sliceOp.getMixedStrides());
                if (!source || failed(offsets) || failed(sizes) || failed(strides)) {
                    return failure();
                }
                return setTensorResult(
                    op, extractSlice(*source, *offsets, *sizes, *strides, sliceOp.getType())
                );
            })
            .Case([&](tensor::InsertSliceOp sliceOp) {
                auto source = getTensor(sliceOp.getSource());
                auto dest = getTensor(sliceOp.getDest());
                auto offsets = getInts(sliceOp.getMixedOffsets());
                auto sizes = getInts(sliceOp.getMixedSizes());
                auto strides = getInts(sliceOp.getMixedStrides());
                if (!source || !dest || failed(offsets) || failed(sizes) || failed(strides)) {
                    return failure();
                }
                return setTensorResult(op, insertSlice(*source, dest, *offsets, *sizes, *strides));
            })
            .Case<tensor::ExpandShapeOp, tensor::CollapseShapeOp>([&](auto reshapeOp) {
                auto source = getTensor(reshapeOp.getSrc());
                if (!source) {
                    return failure();
                }
                return setTensorResult(op, reshape(*source, reshapeOp.getType()));
            })
            .Case([&](torq_hl::TransposeOp transposeOp) {
                auto input = getTensor(transposeOp.getInput());
                if (!input || transposeOp->getNumResults() != 1) {
                    return failure();
                }
                return setTensorResult(
                    op, transpose(*input, transposeOp.getPerm(), op->getResult(0).getType())
                );
            })
            .Case([&](linalg::TransposeOp transposeOp) {
                auto input = getTensor(transposeOp.getInput());
                if (!input || transposeOp->getNumResults() != 1) {
                    return failure();
                }
                return setTensorResult(
                    op, transpose(*input, transposeOp.getPermutation(), op->getResult(0).getType())
                );
            })
            .Case<linalg::GenericOp, linalg::ReduceOp>([&](auto linalgOp) {
                return evalLinalgOp(cast<linalg::LinalgOp>(linalgOp.getOperation()));
            })
            .Case([&](scf::ForOp forOp) {
                auto result = evalFor(forOp);
                forgetInnerValues(forOp);
                return result;
            })
            .Case([&](scf::ForallOp forallOp) {
                auto result = evalForall(forallOp);
                forgetInnerValues(forallOp);
                return result;
            })
            .Default([&](Operation *op) {
                SmallVector<Attribute> operands;
                for (Value operand : op->getOperands()) {
                    auto eval = get(operand);
                    if (failed(eval) || !eval->scalar) {
                        return failure();
                    }
                    operands.push_back(eval->scalar);
                }
                return foldScalarOp(op, operands);
            });
    }

    bool recursive;
    ArrayRef<Value> assumeZero;
    Operation *rootOp = nullptr;
    DenseMap<Value, Eval> env;
};

} // namespace

class ConstantEvaluator::Impl : public Evaluator {
    using Evaluator::Evaluator;
};

ConstantEvaluator::ConstantEvaluator(bool recursive, llvm::ArrayRef<Value> assumeZero)
    : impl(std::make_unique<Impl>(recursive, assumeZero)) {}

ConstantEvaluator::~ConstantEvaluator() = default;

DenseElementsAttr ConstantEvaluator::evaluate(Value value) { return impl->evaluate(value); }

} // namespace mlir::syna::torq
