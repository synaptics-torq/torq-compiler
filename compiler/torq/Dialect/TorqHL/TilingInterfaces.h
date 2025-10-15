#pragma once

namespace mlir {
class DialectRegistry;

namespace syna::torq_hl {
void registerTilingInterfaceExternalModels(DialectRegistry &registry);
} // namespace syna::torq_hl
} // namespace mlir
