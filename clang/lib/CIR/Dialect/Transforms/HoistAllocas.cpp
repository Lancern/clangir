//====- HoistAllocas.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

#include "llvm/Support/TimeProfiler.h"

using namespace mlir;
using namespace cir;

namespace {

struct HoistAllocasPass : public HoistAllocasBase<HoistAllocasPass> {

  HoistAllocasPass() = default;
  void runOnOperation() override;
};

static bool isOpInLoop(mlir::Operation *op) {
  return op->getParentOfType<cir::LoopOpInterface>();
}

static bool isWhileCondition(cir::AllocaOp alloca) {
  for (mlir::Operation *user : alloca->getUsers()) {
    if (!mlir::isa<cir::StoreOp>(user))
      continue;

    auto store = mlir::cast<cir::StoreOp>(user);
    mlir::Operation *storeParentOp = store->getParentOp();
    if (!mlir::isa<cir::WhileOp>(storeParentOp))
      continue;

    auto whileOp = mlir::cast<cir::WhileOp>(storeParentOp);
    return &whileOp.getCond() == store->getParentRegion();
  }

  return false;
}

static void process(cir::FuncOp func) {
  if (func.getRegion().empty())
    return;

  // Hoist all static allocas to the entry block.
  mlir::Block &entryBlock = func.getRegion().front();
  llvm::SmallVector<cir::AllocaOp> allocas;
  func.getBody().walk([&](cir::AllocaOp alloca) {
    if (alloca->getBlock() == &entryBlock)
      return;
    // Don't hoist allocas with dynamic alloca size.
    if (alloca.getDynAllocSize())
      return;
    allocas.push_back(alloca);
  });
  if (allocas.empty())
    return;

  mlir::Operation *insertPoint = &*entryBlock.begin();

  for (auto alloca : allocas) {
    if (alloca.getConstant()) {
      if (isOpInLoop(alloca)) {
        mlir::OpBuilder builder(alloca);
        auto invariantGroupOp =
            builder.create<cir::InvariantGroupOp>(alloca.getLoc(), alloca);
        alloca->replaceUsesWithIf(
            invariantGroupOp,
            [op = invariantGroupOp.getOperation()](mlir::OpOperand &use) {
              return use.getOwner() != op;
            });
      } else if (isWhileCondition(alloca)) {
        // The alloca represents a variable declared as the condition of a while
        // loop. In CIR, the alloca would be emitted at a scope outside of the
        // while loop. We have to remove the constant flag during hoisting,
        // otherwise we would be telling the optimizer that the alloca-ed value
        // is constant across all iterations of the while loop.
        alloca.setConstant(false);
      }
    }

    alloca->moveBefore(insertPoint);
  }
}

void HoistAllocasPass::runOnOperation() {
  llvm::TimeTraceScope scope("Hoist Allocas");
  llvm::SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](cir::FuncOp op) { process(op); });
}

} // namespace

std::unique_ptr<Pass> mlir::createHoistAllocasPass() {
  return std::make_unique<HoistAllocasPass>();
}