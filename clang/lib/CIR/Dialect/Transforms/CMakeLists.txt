add_subdirectory(TargetLowering)

add_clang_library(MLIRCIRTransforms
  LifetimeCheck.cpp
  LoweringPrepare.cpp
  CIRCanonicalize.cpp
  CIRSimplify.cpp
  DropAST.cpp
  IdiomRecognizer.cpp
  LibOpt.cpp
  StdHelpers.cpp
  FlattenCFG.cpp
  GotoSolver.cpp
  SCFPrepare.cpp
  ABILowering.cpp
  CallConvLowering.cpp
  HoistAllocas.cpp
  PointsToAnalysis.cpp
  LiveObjectAnalysis.cpp
  MoveOpt.cpp
  StdLibStatistics.cpp

  DEPENDS
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC
  clangAST
  clangBasic
  clangFrontend
  TargetLowering

  MLIRAnalysis
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRSideEffectInterfaces

  MLIRCIR
  MLIRCIRInterfaces
)
