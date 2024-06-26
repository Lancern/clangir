; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 2
; RUN: opt -passes='loop-mssa(licm)' -S < %s | FileCheck %s --check-prefixes=CHECK,NOT_CONSTRAINED
; RUN: opt -passes='loop-mssa(licm)' -licm-max-num-int-reassociations=1 -S < %s | FileCheck %s --check-prefixes=CHECK,CONSTRAINED

;
; A simple loop:
;
;  int j;
;
;  for (j = 0; j <= i; j++)
;    cells[j] = d1 * cells[j + 1] * delta;
;
; ...should be transformed by the LICM pass into this:
;
;  int j;
;  const uint64_t d1d = d1 * delta;
;
;  for (j = 0; j <= i; j++)
;    cells[j] = d1d * cells[j + 1];
;

define void @innermost_loop_1d_shouldhoist(i32 %i, i64 %d1, i64 %delta, ptr %cells) {
; CHECK-LABEL: define void @innermost_loop_1d_shouldhoist
; CHECK-SAME: (i32 [[I:%.*]], i64 [[D1:%.*]], i64 [[DELTA:%.*]], ptr [[CELLS:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[MUL_1:%.*]] = mul nuw nsw i64 [[DELTA]], [[D1]]
; CHECK-NEXT:    br label [[FOR_COND:%.*]]
; CHECK:       for.cond:
; CHECK-NEXT:    [[J:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[ADD_J_1:%.*]], [[FOR_BODY:%.*]] ]
; CHECK-NEXT:    [[CMP_NOT:%.*]] = icmp sgt i32 [[J]], [[I]]
; CHECK-NEXT:    br i1 [[CMP_NOT]], label [[FOR_END:%.*]], label [[FOR_BODY]]
; CHECK:       for.body:
; CHECK-NEXT:    [[ADD_J_1]] = add nuw nsw i32 [[J]], 1
; CHECK-NEXT:    [[IDXPROM_J_1:%.*]] = zext i32 [[ADD_J_1]] to i64
; CHECK-NEXT:    [[ARRAYIDX_J_1:%.*]] = getelementptr inbounds i64, ptr [[CELLS]], i64 [[IDXPROM_J_1]]
; CHECK-NEXT:    [[CELL_1:%.*]] = load i64, ptr [[ARRAYIDX_J_1]], align 8
; CHECK-NEXT:    [[MUL_2:%.*]] = mul i64 [[MUL_1]], [[CELL_1]]
; CHECK-NEXT:    [[IDXPROM_J:%.*]] = zext i32 [[J]] to i64
; CHECK-NEXT:    [[ARRAYIDX_J:%.*]] = getelementptr inbounds i64, ptr [[CELLS]], i64 [[IDXPROM_J]]
; CHECK-NEXT:    store i64 [[MUL_2]], ptr [[ARRAYIDX_J]], align 8
; CHECK-NEXT:    br label [[FOR_COND]]
; CHECK:       for.end:
; CHECK-NEXT:    ret void
;
entry:
  br label %for.cond

for.cond:
  %j = phi i32 [ 0, %entry ], [ %add.j.1, %for.body ]
  %cmp.not = icmp sgt i32 %j, %i
  br i1 %cmp.not, label %for.end, label %for.body

for.body:
  %add.j.1 = add nuw nsw i32 %j, 1
  %idxprom.j.1 = zext i32 %add.j.1 to i64
  %arrayidx.j.1 = getelementptr inbounds i64, ptr %cells, i64 %idxprom.j.1
  %cell.1 = load i64, ptr %arrayidx.j.1, align 8
  %mul.1 = mul nsw nuw i64 %delta, %d1
  %mul.2 = mul i64 %mul.1, %cell.1
  %idxprom.j = zext i32 %j to i64
  %arrayidx.j = getelementptr inbounds i64, ptr %cells, i64 %idxprom.j
  store i64 %mul.2, ptr %arrayidx.j, align 8
  br label %for.cond

for.end:
  ret void
}

;
; The following loop will be modified by the 'Reassociate expressions' pass,
;
;  int j;
;  const uint64_t d1d = d1 * delta;
;  const uint64_t d2d = d2 * delta;
;
;  for (j = 0; j <= i; j++)
;    cells[j] = d1d * cells[j + 1] + d2d * cells[j];
;
; ...into this:
;
;  int j;
;
;  for (j = 0; j <= i; j++)
;    cells[j] = (d1 * cells[j + 1] + d2 * cells[j]) * delta;
;
; We expect the LICM pass to undo this transformation.
;

define void @innermost_loop_2d(i32 %i, i64 %d1, i64 %d2, i64 %delta, ptr %cells) {
; NOT_CONSTRAINED-LABEL: define void @innermost_loop_2d
; NOT_CONSTRAINED-SAME: (i32 [[I:%.*]], i64 [[D1:%.*]], i64 [[D2:%.*]], i64 [[DELTA:%.*]], ptr [[CELLS:%.*]]) {
; NOT_CONSTRAINED-NEXT:  entry:
; NOT_CONSTRAINED-NEXT:    [[FACTOR_OP_MUL:%.*]] = mul i64 [[D1]], [[DELTA]]
; NOT_CONSTRAINED-NEXT:    [[FACTOR_OP_MUL1:%.*]] = mul i64 [[D2]], [[DELTA]]
; NOT_CONSTRAINED-NEXT:    br label [[FOR_COND:%.*]]
; NOT_CONSTRAINED:       for.cond:
; NOT_CONSTRAINED-NEXT:    [[J:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[ADD_J_1:%.*]], [[FOR_BODY:%.*]] ]
; NOT_CONSTRAINED-NEXT:    [[CMP_NOT:%.*]] = icmp sgt i32 [[J]], [[I]]
; NOT_CONSTRAINED-NEXT:    br i1 [[CMP_NOT]], label [[FOR_END:%.*]], label [[FOR_BODY]]
; NOT_CONSTRAINED:       for.body:
; NOT_CONSTRAINED-NEXT:    [[ADD_J_1]] = add nuw nsw i32 [[J]], 1
; NOT_CONSTRAINED-NEXT:    [[IDXPROM_J_1:%.*]] = zext i32 [[ADD_J_1]] to i64
; NOT_CONSTRAINED-NEXT:    [[ARRAYIDX_J_1:%.*]] = getelementptr inbounds i64, ptr [[CELLS]], i64 [[IDXPROM_J_1]]
; NOT_CONSTRAINED-NEXT:    [[CELL_1:%.*]] = load i64, ptr [[ARRAYIDX_J_1]], align 8
; NOT_CONSTRAINED-NEXT:    [[MUL_1:%.*]] = mul i64 [[CELL_1]], [[FACTOR_OP_MUL]]
; NOT_CONSTRAINED-NEXT:    [[IDXPROM_J:%.*]] = zext i32 [[J]] to i64
; NOT_CONSTRAINED-NEXT:    [[ARRAYIDX_J:%.*]] = getelementptr inbounds i64, ptr [[CELLS]], i64 [[IDXPROM_J]]
; NOT_CONSTRAINED-NEXT:    [[CELL_2:%.*]] = load i64, ptr [[ARRAYIDX_J]], align 8
; NOT_CONSTRAINED-NEXT:    [[MUL_2:%.*]] = mul i64 [[CELL_2]], [[FACTOR_OP_MUL1]]
; NOT_CONSTRAINED-NEXT:    [[REASS_ADD:%.*]] = add i64 [[MUL_2]], [[MUL_1]]
; NOT_CONSTRAINED-NEXT:    store i64 [[REASS_ADD]], ptr [[ARRAYIDX_J]], align 8
; NOT_CONSTRAINED-NEXT:    br label [[FOR_COND]]
; NOT_CONSTRAINED:       for.end:
; NOT_CONSTRAINED-NEXT:    ret void
;
; CONSTRAINED-LABEL: define void @innermost_loop_2d
; CONSTRAINED-SAME: (i32 [[I:%.*]], i64 [[D1:%.*]], i64 [[D2:%.*]], i64 [[DELTA:%.*]], ptr [[CELLS:%.*]]) {
; CONSTRAINED-NEXT:  entry:
; CONSTRAINED-NEXT:    br label [[FOR_COND:%.*]]
; CONSTRAINED:       for.cond:
; CONSTRAINED-NEXT:    [[J:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[ADD_J_1:%.*]], [[FOR_BODY:%.*]] ]
; CONSTRAINED-NEXT:    [[CMP_NOT:%.*]] = icmp sgt i32 [[J]], [[I]]
; CONSTRAINED-NEXT:    br i1 [[CMP_NOT]], label [[FOR_END:%.*]], label [[FOR_BODY]]
; CONSTRAINED:       for.body:
; CONSTRAINED-NEXT:    [[ADD_J_1]] = add nuw nsw i32 [[J]], 1
; CONSTRAINED-NEXT:    [[IDXPROM_J_1:%.*]] = zext i32 [[ADD_J_1]] to i64
; CONSTRAINED-NEXT:    [[ARRAYIDX_J_1:%.*]] = getelementptr inbounds i64, ptr [[CELLS]], i64 [[IDXPROM_J_1]]
; CONSTRAINED-NEXT:    [[CELL_1:%.*]] = load i64, ptr [[ARRAYIDX_J_1]], align 8
; CONSTRAINED-NEXT:    [[MUL_1:%.*]] = mul i64 [[CELL_1]], [[D1]]
; CONSTRAINED-NEXT:    [[IDXPROM_J:%.*]] = zext i32 [[J]] to i64
; CONSTRAINED-NEXT:    [[ARRAYIDX_J:%.*]] = getelementptr inbounds i64, ptr [[CELLS]], i64 [[IDXPROM_J]]
; CONSTRAINED-NEXT:    [[CELL_2:%.*]] = load i64, ptr [[ARRAYIDX_J]], align 8
; CONSTRAINED-NEXT:    [[MUL_2:%.*]] = mul nuw nsw i64 [[CELL_2]], [[D2]]
; CONSTRAINED-NEXT:    [[REASS_ADD:%.*]] = add nuw nsw i64 [[MUL_2]], [[MUL_1]]
; CONSTRAINED-NEXT:    [[REASS_MUL:%.*]] = mul i64 [[REASS_ADD]], [[DELTA]]
; CONSTRAINED-NEXT:    store i64 [[REASS_MUL]], ptr [[ARRAYIDX_J]], align 8
; CONSTRAINED-NEXT:    br label [[FOR_COND]]
; CONSTRAINED:       for.end:
; CONSTRAINED-NEXT:    ret void
;
entry:
  br label %for.cond

for.cond:
  %j = phi i32 [ 0, %entry ], [ %add.j.1, %for.body ]
  %cmp.not = icmp sgt i32 %j, %i
  br i1 %cmp.not, label %for.end, label %for.body

for.body:
  %add.j.1 = add nuw nsw i32 %j, 1
  %idxprom.j.1 = zext i32 %add.j.1 to i64
  %arrayidx.j.1 = getelementptr inbounds i64, ptr %cells, i64 %idxprom.j.1
  %cell.1 = load i64, ptr %arrayidx.j.1, align 8
  %mul.1 = mul i64 %cell.1, %d1
  %idxprom.j = zext i32 %j to i64
  %arrayidx.j = getelementptr inbounds i64, ptr %cells, i64 %idxprom.j
  %cell.2 = load i64, ptr %arrayidx.j, align 8
  %mul.2 = mul nsw nuw i64 %cell.2, %d2
  %reass.add = add nsw nuw i64 %mul.2, %mul.1
  %reass.mul = mul i64 %reass.add, %delta
  store i64 %reass.mul, ptr %arrayidx.j, align 8
  br label %for.cond

for.end:
  ret void
}

;
; The following loop will be modified by the 'Reassociate expressions' pass,
;
;  int j;
;  const uint64_t d1d = d1 * delta;
;  const uint64_t d2d = d2 * delta;
;  const uint64_t d3d = d3 * delta;
;
;  for (j = 0; j <= i; j++)
;    cells[j] = d1d * cells[j + 1] + d2d * cells[j] + d3d * cells[j + 2];
;
; ...into this:
;
;  int j;
;
;  for (j = 0; j <= i; j++)
;    cells[j] = (d1 * cells[j + 1] + d2 * cells[j] + d3 * cells[j + 2]) * delta;
;
; We expect the LICM pass to undo this transformation.
;


define void @innermost_loop_3d(i32 %i, i64 %d1, i64 %d2, i64 %d3, i64 %delta, ptr %cells) {
; NOT_CONSTRAINED-LABEL: define void @innermost_loop_3d
; NOT_CONSTRAINED-SAME: (i32 [[I:%.*]], i64 [[D1:%.*]], i64 [[D2:%.*]], i64 [[D3:%.*]], i64 [[DELTA:%.*]], ptr [[CELLS:%.*]]) {
; NOT_CONSTRAINED-NEXT:  entry:
; NOT_CONSTRAINED-NEXT:    [[FACTOR_OP_MUL:%.*]] = mul i64 [[D3]], [[DELTA]]
; NOT_CONSTRAINED-NEXT:    [[FACTOR_OP_MUL1:%.*]] = mul i64 [[D1]], [[DELTA]]
; NOT_CONSTRAINED-NEXT:    [[FACTOR_OP_MUL2:%.*]] = mul i64 [[D2]], [[DELTA]]
; NOT_CONSTRAINED-NEXT:    br label [[FOR_COND:%.*]]
; NOT_CONSTRAINED:       for.cond:
; NOT_CONSTRAINED-NEXT:    [[J:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[ADD_J_1:%.*]], [[FOR_BODY:%.*]] ]
; NOT_CONSTRAINED-NEXT:    [[CMP_NOT:%.*]] = icmp sgt i32 [[J]], [[I]]
; NOT_CONSTRAINED-NEXT:    br i1 [[CMP_NOT]], label [[FOR_END:%.*]], label [[FOR_BODY]]
; NOT_CONSTRAINED:       for.body:
; NOT_CONSTRAINED-NEXT:    [[ADD_J_1]] = add nuw nsw i32 [[J]], 1
; NOT_CONSTRAINED-NEXT:    [[IDXPROM_J_1:%.*]] = zext i32 [[ADD_J_1]] to i64
; NOT_CONSTRAINED-NEXT:    [[ARRAYIDX_J_1:%.*]] = getelementptr inbounds i64, ptr [[CELLS]], i64 [[IDXPROM_J_1]]
; NOT_CONSTRAINED-NEXT:    [[CELL_1:%.*]] = load i64, ptr [[ARRAYIDX_J_1]], align 8
; NOT_CONSTRAINED-NEXT:    [[MUL_1:%.*]] = mul i64 [[CELL_1]], [[FACTOR_OP_MUL1]]
; NOT_CONSTRAINED-NEXT:    [[IDXPROM_J:%.*]] = zext i32 [[J]] to i64
; NOT_CONSTRAINED-NEXT:    [[ARRAYIDX_J:%.*]] = getelementptr inbounds i64, ptr [[CELLS]], i64 [[IDXPROM_J]]
; NOT_CONSTRAINED-NEXT:    [[CELL_2:%.*]] = load i64, ptr [[ARRAYIDX_J]], align 8
; NOT_CONSTRAINED-NEXT:    [[MUL_2:%.*]] = mul i64 [[CELL_2]], [[FACTOR_OP_MUL2]]
; NOT_CONSTRAINED-NEXT:    [[ADD_J_2:%.*]] = add nuw nsw i32 [[J]], 2
; NOT_CONSTRAINED-NEXT:    [[IDXPROM_J_2:%.*]] = zext i32 [[ADD_J_2]] to i64
; NOT_CONSTRAINED-NEXT:    [[ARRAYIDX_J_2:%.*]] = getelementptr inbounds i64, ptr [[CELLS]], i64 [[IDXPROM_J_2]]
; NOT_CONSTRAINED-NEXT:    [[CELL_3:%.*]] = load i64, ptr [[ARRAYIDX_J_2]], align 8
; NOT_CONSTRAINED-NEXT:    [[MUL_3:%.*]] = mul i64 [[CELL_3]], [[FACTOR_OP_MUL]]
; NOT_CONSTRAINED-NEXT:    [[REASS_ADD:%.*]] = add i64 [[MUL_2]], [[MUL_1]]
; NOT_CONSTRAINED-NEXT:    [[REASS_ADD1:%.*]] = add i64 [[REASS_ADD]], [[MUL_3]]
; NOT_CONSTRAINED-NEXT:    store i64 [[REASS_ADD1]], ptr [[ARRAYIDX_J_2]], align 8
; NOT_CONSTRAINED-NEXT:    br label [[FOR_COND]]
; NOT_CONSTRAINED:       for.end:
; NOT_CONSTRAINED-NEXT:    ret void
;
; CONSTRAINED-LABEL: define void @innermost_loop_3d
; CONSTRAINED-SAME: (i32 [[I:%.*]], i64 [[D1:%.*]], i64 [[D2:%.*]], i64 [[D3:%.*]], i64 [[DELTA:%.*]], ptr [[CELLS:%.*]]) {
; CONSTRAINED-NEXT:  entry:
; CONSTRAINED-NEXT:    br label [[FOR_COND:%.*]]
; CONSTRAINED:       for.cond:
; CONSTRAINED-NEXT:    [[J:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[ADD_J_1:%.*]], [[FOR_BODY:%.*]] ]
; CONSTRAINED-NEXT:    [[CMP_NOT:%.*]] = icmp sgt i32 [[J]], [[I]]
; CONSTRAINED-NEXT:    br i1 [[CMP_NOT]], label [[FOR_END:%.*]], label [[FOR_BODY]]
; CONSTRAINED:       for.body:
; CONSTRAINED-NEXT:    [[ADD_J_1]] = add nuw nsw i32 [[J]], 1
; CONSTRAINED-NEXT:    [[IDXPROM_J_1:%.*]] = zext i32 [[ADD_J_1]] to i64
; CONSTRAINED-NEXT:    [[ARRAYIDX_J_1:%.*]] = getelementptr inbounds i64, ptr [[CELLS]], i64 [[IDXPROM_J_1]]
; CONSTRAINED-NEXT:    [[CELL_1:%.*]] = load i64, ptr [[ARRAYIDX_J_1]], align 8
; CONSTRAINED-NEXT:    [[MUL_1:%.*]] = mul i64 [[CELL_1]], [[D1]]
; CONSTRAINED-NEXT:    [[IDXPROM_J:%.*]] = zext i32 [[J]] to i64
; CONSTRAINED-NEXT:    [[ARRAYIDX_J:%.*]] = getelementptr inbounds i64, ptr [[CELLS]], i64 [[IDXPROM_J]]
; CONSTRAINED-NEXT:    [[CELL_2:%.*]] = load i64, ptr [[ARRAYIDX_J]], align 8
; CONSTRAINED-NEXT:    [[MUL_2:%.*]] = mul i64 [[CELL_2]], [[D2]]
; CONSTRAINED-NEXT:    [[ADD_J_2:%.*]] = add nuw nsw i32 [[J]], 2
; CONSTRAINED-NEXT:    [[IDXPROM_J_2:%.*]] = zext i32 [[ADD_J_2]] to i64
; CONSTRAINED-NEXT:    [[ARRAYIDX_J_2:%.*]] = getelementptr inbounds i64, ptr [[CELLS]], i64 [[IDXPROM_J_2]]
; CONSTRAINED-NEXT:    [[CELL_3:%.*]] = load i64, ptr [[ARRAYIDX_J_2]], align 8
; CONSTRAINED-NEXT:    [[MUL_3:%.*]] = mul nuw nsw i64 [[CELL_3]], [[D3]]
; CONSTRAINED-NEXT:    [[REASS_ADD:%.*]] = add nuw nsw i64 [[MUL_2]], [[MUL_1]]
; CONSTRAINED-NEXT:    [[REASS_ADD1:%.*]] = add nuw nsw i64 [[REASS_ADD]], [[MUL_3]]
; CONSTRAINED-NEXT:    [[REASS_MUL:%.*]] = mul nuw nsw i64 [[REASS_ADD1]], [[DELTA]]
; CONSTRAINED-NEXT:    store i64 [[REASS_MUL]], ptr [[ARRAYIDX_J_2]], align 8
; CONSTRAINED-NEXT:    br label [[FOR_COND]]
; CONSTRAINED:       for.end:
; CONSTRAINED-NEXT:    ret void
;
entry:
  br label %for.cond

for.cond:
  %j = phi i32 [ 0, %entry ], [ %add.j.1, %for.body ]
  %cmp.not = icmp sgt i32 %j, %i
  br i1 %cmp.not, label %for.end, label %for.body

for.body:
  %add.j.1 = add nuw nsw i32 %j, 1
  %idxprom.j.1 = zext i32 %add.j.1 to i64
  %arrayidx.j.1 = getelementptr inbounds i64, ptr %cells, i64 %idxprom.j.1
  %cell.1 = load i64, ptr %arrayidx.j.1, align 8
  %mul.1 = mul i64 %cell.1, %d1
  %idxprom.j = zext i32 %j to i64
  %arrayidx.j = getelementptr inbounds i64, ptr %cells, i64 %idxprom.j
  %cell.2 = load i64, ptr %arrayidx.j, align 8
  %mul.2 = mul i64 %cell.2, %d2
  %add.j.2 = add nuw nsw i32 %j, 2
  %idxprom.j.2 = zext i32 %add.j.2 to i64
  %arrayidx.j.2 = getelementptr inbounds i64, ptr %cells, i64 %idxprom.j.2
  %cell.3 = load i64, ptr %arrayidx.j.2, align 8
  %mul.3 = mul nsw nuw i64 %cell.3, %d3
  %reass.add = add nsw nuw i64 %mul.2, %mul.1
  %reass.add1 = add nsw nuw i64 %reass.add, %mul.3
  %reass.mul = mul nsw nuw i64 %reass.add1, %delta
  store i64 %reass.mul, ptr %arrayidx.j.2, align 8
  br label %for.cond

for.end:
  ret void
}

;
; The following loop will not be modified by the LICM pass:
;
;  int j;
;
;  for (j = 0; j <= i; j++)
;    cells[j] = (d1 * cells[j + 1] + d2 * cells[j] +
;                cells[j] * cells[j + 1]) * delta;
;
; This case differs as one of the multiplications involves no invariants.
;

define void @innermost_loop_3d_reassociated_different(i32 %i, i64 %d1, i64 %d2, i64 %delta, ptr %cells) {
; CHECK-LABEL: define void @innermost_loop_3d_reassociated_different
; CHECK-SAME: (i32 [[I:%.*]], i64 [[D1:%.*]], i64 [[D2:%.*]], i64 [[DELTA:%.*]], ptr [[CELLS:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[FOR_COND:%.*]]
; CHECK:       for.cond:
; CHECK-NEXT:    [[J:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[ADD_J_1:%.*]], [[FOR_BODY:%.*]] ]
; CHECK-NEXT:    [[CMP_NOT:%.*]] = icmp sgt i32 [[J]], [[I]]
; CHECK-NEXT:    br i1 [[CMP_NOT]], label [[FOR_END:%.*]], label [[FOR_BODY]]
; CHECK:       for.body:
; CHECK-NEXT:    [[ADD_J_1]] = add nuw nsw i32 [[J]], 1
; CHECK-NEXT:    [[IDXPROM_J_1:%.*]] = zext i32 [[ADD_J_1]] to i64
; CHECK-NEXT:    [[ARRAYIDX_J_1:%.*]] = getelementptr inbounds i64, ptr [[CELLS]], i64 [[IDXPROM_J_1]]
; CHECK-NEXT:    [[CELL_1:%.*]] = load i64, ptr [[ARRAYIDX_J_1]], align 8
; CHECK-NEXT:    [[IDXPROM_J_2:%.*]] = zext i32 [[ADD_J_1]] to i64
; CHECK-NEXT:    [[ARRAYIDX_J_2:%.*]] = getelementptr inbounds i64, ptr [[CELLS]], i64 [[IDXPROM_J_2]]
; CHECK-NEXT:    [[CELL_2:%.*]] = load i64, ptr [[ARRAYIDX_J_2]], align 8
; CHECK-NEXT:    [[CELL_3:%.*]] = load i64, ptr [[ARRAYIDX_J_2]], align 8
; CHECK-NEXT:    [[IDXPROM_J:%.*]] = zext i32 [[J]] to i64
; CHECK-NEXT:    [[ARRAYIDX_J:%.*]] = getelementptr inbounds i64, ptr [[CELLS]], i64 [[IDXPROM_J]]
; CHECK-NEXT:    [[CELL_4:%.*]] = load i64, ptr [[ARRAYIDX_J]], align 8
; CHECK-NEXT:    [[MUL_1:%.*]] = mul i64 [[CELL_1]], [[D1]]
; CHECK-NEXT:    [[MUL_2:%.*]] = mul i64 [[CELL_4]], [[D2]]
; CHECK-NEXT:    [[EXTRA_MUL:%.*]] = mul i64 [[CELL_3]], [[CELL_2]]
; CHECK-NEXT:    [[REASS_ADD:%.*]] = add i64 [[EXTRA_MUL]], [[MUL_1]]
; CHECK-NEXT:    [[EXTRA_ADD:%.*]] = add i64 [[REASS_ADD]], [[MUL_2]]
; CHECK-NEXT:    [[REASS_MUL:%.*]] = mul i64 [[EXTRA_ADD]], [[DELTA]]
; CHECK-NEXT:    store i64 [[REASS_MUL]], ptr [[ARRAYIDX_J]], align 8
; CHECK-NEXT:    br label [[FOR_COND]]
; CHECK:       for.end:
; CHECK-NEXT:    ret void
;
entry:
  br label %for.cond

for.cond:
  %j = phi i32 [ 0, %entry ], [ %add.j.1, %for.body ]
  %cmp.not = icmp sgt i32 %j, %i
  br i1 %cmp.not, label %for.end, label %for.body

for.body:
  %add.j.1 = add nuw nsw i32 %j, 1
  %idxprom.j.1 = zext i32 %add.j.1 to i64
  %arrayidx.j.1 = getelementptr inbounds i64, ptr %cells, i64 %idxprom.j.1
  %cell.1 = load i64, ptr %arrayidx.j.1, align 8
  %idxprom.j.2 = zext i32 %add.j.1 to i64
  %arrayidx.j.2 = getelementptr inbounds i64, ptr %cells, i64 %idxprom.j.2
  %cell.2 = load i64, ptr %arrayidx.j.2, align 8
  %idxprom.j.3 = zext i32 %add.j.1 to i64
  %cell.3 = load i64, ptr %arrayidx.j.2, align 8
  %idxprom.j = zext i32 %j to i64
  %arrayidx.j = getelementptr inbounds i64, ptr %cells, i64 %idxprom.j
  %cell.4 = load i64, ptr %arrayidx.j, align 8
  %mul.1 = mul i64 %cell.1, %d1
  %mul.2 = mul i64 %cell.4, %d2
  %extra.mul = mul i64 %cell.3, %cell.2
  %reass.add = add i64 %extra.mul, %mul.1
  %extra.add = add i64 %reass.add, %mul.2
  %reass.mul = mul i64 %extra.add, %delta
  store i64 %reass.mul, ptr %arrayidx.j, align 8
  br label %for.cond

for.end:
  ret void
}

; Make sure we drop poison flags on the mul in the loop.
define i32 @pr85457(i32 %x, i32 %y) {
; CHECK-LABEL: define i32 @pr85457
; CHECK-SAME: (i32 [[X:%.*]], i32 [[Y:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[FACTOR_OP_MUL:%.*]] = mul i32 [[X]], [[Y]]
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ 1, [[ENTRY:%.*]] ], [ [[IV_NEXT:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[IV_NEXT]] = add nuw nsw i32 [[IV]], 1
; CHECK-NEXT:    [[MUL0:%.*]] = mul i32 [[FACTOR_OP_MUL]], [[IV]]
; CHECK-NEXT:    [[CMP:%.*]] = icmp slt i32 [[MUL0]], 1
; CHECK-NEXT:    br i1 [[CMP]], label [[EXIT:%.*]], label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    ret i32 0
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ 1, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i32 %iv, 1
  %mul0 = mul nuw nsw i32 %x, %iv
  %mul1 = mul nuw i32 %mul0, %y
  %cmp = icmp slt i32 %mul1, 1
  br i1 %cmp, label %exit, label %loop

exit:
  ret i32 0
}
