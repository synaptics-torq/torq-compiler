#ifndef TORQ_HL_TRAITS_H
#define TORQ_HL_TRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::OpTrait {

template <typename ConcreteType>
class UpdateInPlaceTrait : public TraitBase<ConcreteType, UpdateInPlaceTrait> {};
} // namespace mlir::OpTrait
#endif // TORQ_HL_TRAITS_H