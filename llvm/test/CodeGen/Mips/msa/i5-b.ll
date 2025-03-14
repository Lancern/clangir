; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; Test the MSA intrinsics that are encoded with the I5 instruction format.
; There are lots of these so this covers those beginning with 'b'

; RUN: llc -mtriple=mips -mattr=+msa,+fp64,+mips32r2 -relocation-model=pic < %s | FileCheck %s
; RUN: llc -mtriple=mipsel -mattr=+msa,+fp64,+mips32r2 -relocation-model=pic < %s | FileCheck %s

@llvm_mips_bclri_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bclri_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bclri_b_test() nounwind {
; CHECK-LABEL: llvm_mips_bclri_b_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_bclri_b_ARG1)($1)
; CHECK-NEXT:    ld.b $w0, 0($2)
; CHECK-NEXT:    andi.b $w0, $w0, 127
; CHECK-NEXT:    lw $1, %got(llvm_mips_bclri_b_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.b $w0, 0($1)
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_bclri_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.bclri.b(<16 x i8> %0, i32 7)
  store <16 x i8> %1, ptr @llvm_mips_bclri_b_RES
  ret void
}
declare <16 x i8> @llvm.mips.bclri.b(<16 x i8>, i32) nounwind

@llvm_mips_bclri_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_bclri_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_bclri_h_test() nounwind {
; CHECK-LABEL: llvm_mips_bclri_h_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_bclri_h_ARG1)($1)
; CHECK-NEXT:    ld.h $w0, 0($2)
; CHECK-NEXT:    bclri.h $w0, $w0, 7
; CHECK-NEXT:    lw $1, %got(llvm_mips_bclri_h_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.h $w0, 0($1)
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_bclri_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.bclri.h(<8 x i16> %0, i32 7)
  store <8 x i16> %1, ptr @llvm_mips_bclri_h_RES
  ret void
}
declare <8 x i16> @llvm.mips.bclri.h(<8 x i16>, i32) nounwind

@llvm_mips_bclri_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_bclri_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_bclri_w_test() nounwind {
; CHECK-LABEL: llvm_mips_bclri_w_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_bclri_w_ARG1)($1)
; CHECK-NEXT:    ld.w $w0, 0($2)
; CHECK-NEXT:    bclri.w $w0, $w0, 7
; CHECK-NEXT:    lw $1, %got(llvm_mips_bclri_w_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.w $w0, 0($1)
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_bclri_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.bclri.w(<4 x i32> %0, i32 7)
  store <4 x i32> %1, ptr @llvm_mips_bclri_w_RES
  ret void
}
declare <4 x i32> @llvm.mips.bclri.w(<4 x i32>, i32) nounwind

@llvm_mips_bclri_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_bclri_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_bclri_d_test() nounwind {
; CHECK-LABEL: llvm_mips_bclri_d_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_bclri_d_ARG1)($1)
; CHECK-NEXT:    ld.d $w0, 0($2)
; CHECK-NEXT:    bclri.d $w0, $w0, 7
; CHECK-NEXT:    lw $1, %got(llvm_mips_bclri_d_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.d $w0, 0($1)
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_bclri_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.bclri.d(<2 x i64> %0, i32 7)
  store <2 x i64> %1, ptr @llvm_mips_bclri_d_RES
  ret void
}
declare <2 x i64> @llvm.mips.bclri.d(<2 x i64>, i32) nounwind

@llvm_mips_binsli_b_ARG1 = global <16 x i8> zeroinitializer, align 16
@llvm_mips_binsli_b_ARG2 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_binsli_b_RES  = global <16 x i8> zeroinitializer, align 16

define void @llvm_mips_binsli_b_test() nounwind {
; CHECK-LABEL: llvm_mips_binsli_b_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_binsli_b_ARG1)($1)
; CHECK-NEXT:    lw $3, %got(llvm_mips_binsli_b_ARG2)($1)
; CHECK-NEXT:    ld.b $w0, 0($3)
; CHECK-NEXT:    ld.b $w1, 0($2)
; CHECK-NEXT:    binsli.b $w1, $w0, 6
; CHECK-NEXT:    lw $1, %got(llvm_mips_binsli_b_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.b $w1, 0($1)
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_binsli_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_binsli_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.binsli.b(<16 x i8> %0, <16 x i8> %1, i32 6)
  store <16 x i8> %2, ptr @llvm_mips_binsli_b_RES
  ret void
}
declare <16 x i8> @llvm.mips.binsli.b(<16 x i8>, <16 x i8>, i32) nounwind

@llvm_mips_binsli_h_ARG1 = global <8 x i16> zeroinitializer, align 16
@llvm_mips_binsli_h_ARG2 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_binsli_h_RES  = global <8 x i16> zeroinitializer, align 16

define void @llvm_mips_binsli_h_test() nounwind {
; CHECK-LABEL: llvm_mips_binsli_h_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_binsli_h_ARG1)($1)
; CHECK-NEXT:    lw $3, %got(llvm_mips_binsli_h_ARG2)($1)
; CHECK-NEXT:    ld.h $w0, 0($3)
; CHECK-NEXT:    ld.h $w1, 0($2)
; CHECK-NEXT:    binsli.h $w1, $w0, 7
; CHECK-NEXT:    lw $1, %got(llvm_mips_binsli_h_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.h $w1, 0($1)
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_binsli_h_ARG1
  %1 = load <8 x i16>, ptr @llvm_mips_binsli_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.binsli.h(<8 x i16> %0, <8 x i16> %1, i32 7)
  store <8 x i16> %2, ptr @llvm_mips_binsli_h_RES
  ret void
}
declare <8 x i16> @llvm.mips.binsli.h(<8 x i16>, <8 x i16>, i32) nounwind

@llvm_mips_binsli_w_ARG1 = global <4 x i32> zeroinitializer, align 16
@llvm_mips_binsli_w_ARG2 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_binsli_w_RES  = global <4 x i32> zeroinitializer, align 16

define void @llvm_mips_binsli_w_test() nounwind {
; CHECK-LABEL: llvm_mips_binsli_w_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_binsli_w_ARG1)($1)
; CHECK-NEXT:    lw $3, %got(llvm_mips_binsli_w_ARG2)($1)
; CHECK-NEXT:    ld.w $w0, 0($3)
; CHECK-NEXT:    ld.w $w1, 0($2)
; CHECK-NEXT:    binsli.w $w1, $w0, 7
; CHECK-NEXT:    lw $1, %got(llvm_mips_binsli_w_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.w $w1, 0($1)
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_binsli_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_binsli_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.binsli.w(<4 x i32> %0, <4 x i32> %1, i32 7)
  store <4 x i32> %2, ptr @llvm_mips_binsli_w_RES
  ret void
}
declare <4 x i32> @llvm.mips.binsli.w(<4 x i32>, <4 x i32>, i32) nounwind

@llvm_mips_binsli_d_ARG1 = global <2 x i64> zeroinitializer, align 16
@llvm_mips_binsli_d_ARG2 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_binsli_d_RES  = global <2 x i64> zeroinitializer, align 16

define void @llvm_mips_binsli_d_test() nounwind {
; CHECK-LABEL: llvm_mips_binsli_d_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_binsli_d_ARG1)($1)
; CHECK-NEXT:    lw $3, %got(llvm_mips_binsli_d_ARG2)($1)
; CHECK-NEXT:    ld.d $w0, 0($3)
; CHECK-NEXT:    ld.d $w1, 0($2)
; CHECK-NEXT:    binsli.d $w1, $w0, 61
; CHECK-NEXT:    lw $1, %got(llvm_mips_binsli_d_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.d $w1, 0($1)
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_binsli_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_binsli_d_ARG2
  ; TODO: We use a particularly wide mask here to work around a legalization
  ;       issue. If the mask doesn't fit within a 10-bit immediate, it gets
  ;       legalized into a constant pool. We should add a test to cover the
  ;       other cases once they correctly select binsli.d.
  %2 = tail call <2 x i64> @llvm.mips.binsli.d(<2 x i64> %0, <2 x i64> %1, i32 61)
  store <2 x i64> %2, ptr @llvm_mips_binsli_d_RES
  ret void
}
declare <2 x i64> @llvm.mips.binsli.d(<2 x i64>, <2 x i64>, i32) nounwind

@llvm_mips_binsri_b_ARG1 = global <16 x i8> zeroinitializer, align 16
@llvm_mips_binsri_b_ARG2 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_binsri_b_RES  = global <16 x i8> zeroinitializer, align 16

define void @llvm_mips_binsri_b_test() nounwind {
; CHECK-LABEL: llvm_mips_binsri_b_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_binsri_b_ARG1)($1)
; CHECK-NEXT:    lw $3, %got(llvm_mips_binsri_b_ARG2)($1)
; CHECK-NEXT:    ld.b $w0, 0($3)
; CHECK-NEXT:    ld.b $w1, 0($2)
; CHECK-NEXT:    binsri.b $w1, $w0, 6
; CHECK-NEXT:    lw $1, %got(llvm_mips_binsri_b_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.b $w1, 0($1)
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_binsri_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_binsri_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.binsri.b(<16 x i8> %0, <16 x i8> %1, i32 6)
  store <16 x i8> %2, ptr @llvm_mips_binsri_b_RES
  ret void
}
declare <16 x i8> @llvm.mips.binsri.b(<16 x i8>, <16 x i8>, i32) nounwind

@llvm_mips_binsri_h_ARG1 = global <8 x i16> zeroinitializer, align 16
@llvm_mips_binsri_h_ARG2 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_binsri_h_RES  = global <8 x i16> zeroinitializer, align 16

define void @llvm_mips_binsri_h_test() nounwind {
; CHECK-LABEL: llvm_mips_binsri_h_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_binsri_h_ARG1)($1)
; CHECK-NEXT:    lw $3, %got(llvm_mips_binsri_h_ARG2)($1)
; CHECK-NEXT:    ld.h $w0, 0($3)
; CHECK-NEXT:    ld.h $w1, 0($2)
; CHECK-NEXT:    binsri.h $w1, $w0, 7
; CHECK-NEXT:    lw $1, %got(llvm_mips_binsri_h_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.h $w1, 0($1)
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_binsri_h_ARG1
  %1 = load <8 x i16>, ptr @llvm_mips_binsri_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.binsri.h(<8 x i16> %0, <8 x i16> %1, i32 7)
  store <8 x i16> %2, ptr @llvm_mips_binsri_h_RES
  ret void
}
declare <8 x i16> @llvm.mips.binsri.h(<8 x i16>, <8 x i16>, i32) nounwind

@llvm_mips_binsri_w_ARG1 = global <4 x i32> zeroinitializer, align 16
@llvm_mips_binsri_w_ARG2 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_binsri_w_RES  = global <4 x i32> zeroinitializer, align 16

define void @llvm_mips_binsri_w_test() nounwind {
; CHECK-LABEL: llvm_mips_binsri_w_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_binsri_w_ARG1)($1)
; CHECK-NEXT:    lw $3, %got(llvm_mips_binsri_w_ARG2)($1)
; CHECK-NEXT:    ld.w $w0, 0($3)
; CHECK-NEXT:    ld.w $w1, 0($2)
; CHECK-NEXT:    binsri.w $w1, $w0, 7
; CHECK-NEXT:    lw $1, %got(llvm_mips_binsri_w_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.w $w1, 0($1)
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_binsri_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_binsri_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.binsri.w(<4 x i32> %0, <4 x i32> %1, i32 7)
  store <4 x i32> %2, ptr @llvm_mips_binsri_w_RES
  ret void
}
declare <4 x i32> @llvm.mips.binsri.w(<4 x i32>, <4 x i32>, i32) nounwind

@llvm_mips_binsri_d_ARG1 = global <2 x i64> zeroinitializer, align 16
@llvm_mips_binsri_d_ARG2 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_binsri_d_RES  = global <2 x i64> zeroinitializer, align 16

define void @llvm_mips_binsri_d_test() nounwind {
; CHECK-LABEL: llvm_mips_binsri_d_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_binsri_d_ARG1)($1)
; CHECK-NEXT:    lw $3, %got(llvm_mips_binsri_d_ARG2)($1)
; CHECK-NEXT:    ld.d $w0, 0($3)
; CHECK-NEXT:    ld.d $w1, 0($2)
; CHECK-NEXT:    binsri.d $w1, $w0, 7
; CHECK-NEXT:    lw $1, %got(llvm_mips_binsri_d_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.d $w1, 0($1)
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_binsri_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_binsri_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.binsri.d(<2 x i64> %0, <2 x i64> %1, i32 7)
  store <2 x i64> %2, ptr @llvm_mips_binsri_d_RES
  ret void
}
declare <2 x i64> @llvm.mips.binsri.d(<2 x i64>, <2 x i64>, i32) nounwind

@llvm_mips_bnegi_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bnegi_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bnegi_b_test() nounwind {
; CHECK-LABEL: llvm_mips_bnegi_b_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_bnegi_b_ARG1)($1)
; CHECK-NEXT:    ld.b $w0, 0($2)
; CHECK-NEXT:    bnegi.b $w0, $w0, 7
; CHECK-NEXT:    lw $1, %got(llvm_mips_bnegi_b_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.b $w0, 0($1)
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_bnegi_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.bnegi.b(<16 x i8> %0, i32 7)
  store <16 x i8> %1, ptr @llvm_mips_bnegi_b_RES
  ret void
}
declare <16 x i8> @llvm.mips.bnegi.b(<16 x i8>, i32) nounwind

@llvm_mips_bnegi_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_bnegi_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_bnegi_h_test() nounwind {
; CHECK-LABEL: llvm_mips_bnegi_h_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_bnegi_h_ARG1)($1)
; CHECK-NEXT:    ld.h $w0, 0($2)
; CHECK-NEXT:    bnegi.h $w0, $w0, 7
; CHECK-NEXT:    lw $1, %got(llvm_mips_bnegi_h_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.h $w0, 0($1)
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_bnegi_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.bnegi.h(<8 x i16> %0, i32 7)
  store <8 x i16> %1, ptr @llvm_mips_bnegi_h_RES
  ret void
}
declare <8 x i16> @llvm.mips.bnegi.h(<8 x i16>, i32) nounwind

@llvm_mips_bnegi_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_bnegi_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_bnegi_w_test() nounwind {
; CHECK-LABEL: llvm_mips_bnegi_w_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_bnegi_w_ARG1)($1)
; CHECK-NEXT:    ld.w $w0, 0($2)
; CHECK-NEXT:    bnegi.w $w0, $w0, 7
; CHECK-NEXT:    lw $1, %got(llvm_mips_bnegi_w_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.w $w0, 0($1)
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_bnegi_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.bnegi.w(<4 x i32> %0, i32 7)
  store <4 x i32> %1, ptr @llvm_mips_bnegi_w_RES
  ret void
}
declare <4 x i32> @llvm.mips.bnegi.w(<4 x i32>, i32) nounwind

@llvm_mips_bnegi_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_bnegi_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_bnegi_d_test() nounwind {
; CHECK-LABEL: llvm_mips_bnegi_d_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_bnegi_d_ARG1)($1)
; CHECK-NEXT:    ld.d $w0, 0($2)
; CHECK-NEXT:    bnegi.d $w0, $w0, 7
; CHECK-NEXT:    lw $1, %got(llvm_mips_bnegi_d_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.d $w0, 0($1)
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_bnegi_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.bnegi.d(<2 x i64> %0, i32 7)
  store <2 x i64> %1, ptr @llvm_mips_bnegi_d_RES
  ret void
}
declare <2 x i64> @llvm.mips.bnegi.d(<2 x i64>, i32) nounwind

@llvm_mips_bseti_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bseti_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bseti_b_test() nounwind {
; CHECK-LABEL: llvm_mips_bseti_b_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_bseti_b_ARG1)($1)
; CHECK-NEXT:    ld.b $w0, 0($2)
; CHECK-NEXT:    bseti.b $w0, $w0, 7
; CHECK-NEXT:    lw $1, %got(llvm_mips_bseti_b_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.b $w0, 0($1)
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_bseti_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.bseti.b(<16 x i8> %0, i32 7)
  store <16 x i8> %1, ptr @llvm_mips_bseti_b_RES
  ret void
}
declare <16 x i8> @llvm.mips.bseti.b(<16 x i8>, i32) nounwind

@llvm_mips_bseti_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_bseti_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_bseti_h_test() nounwind {
; CHECK-LABEL: llvm_mips_bseti_h_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_bseti_h_ARG1)($1)
; CHECK-NEXT:    ld.h $w0, 0($2)
; CHECK-NEXT:    bseti.h $w0, $w0, 7
; CHECK-NEXT:    lw $1, %got(llvm_mips_bseti_h_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.h $w0, 0($1)
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_bseti_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.bseti.h(<8 x i16> %0, i32 7)
  store <8 x i16> %1, ptr @llvm_mips_bseti_h_RES
  ret void
}
declare <8 x i16> @llvm.mips.bseti.h(<8 x i16>, i32) nounwind

@llvm_mips_bseti_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_bseti_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_bseti_w_test() nounwind {
; CHECK-LABEL: llvm_mips_bseti_w_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_bseti_w_ARG1)($1)
; CHECK-NEXT:    ld.w $w0, 0($2)
; CHECK-NEXT:    bseti.w $w0, $w0, 7
; CHECK-NEXT:    lw $1, %got(llvm_mips_bseti_w_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.w $w0, 0($1)
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_bseti_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.bseti.w(<4 x i32> %0, i32 7)
  store <4 x i32> %1, ptr @llvm_mips_bseti_w_RES
  ret void
}
declare <4 x i32> @llvm.mips.bseti.w(<4 x i32>, i32) nounwind

@llvm_mips_bseti_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_bseti_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_bseti_d_test() nounwind {
; CHECK-LABEL: llvm_mips_bseti_d_test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lui $2, %hi(_gp_disp)
; CHECK-NEXT:    addiu $2, $2, %lo(_gp_disp)
; CHECK-NEXT:    addu $1, $2, $25
; CHECK-NEXT:    lw $2, %got(llvm_mips_bseti_d_ARG1)($1)
; CHECK-NEXT:    ld.d $w0, 0($2)
; CHECK-NEXT:    bseti.d $w0, $w0, 7
; CHECK-NEXT:    lw $1, %got(llvm_mips_bseti_d_RES)($1)
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    st.d $w0, 0($1)
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_bseti_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.bseti.d(<2 x i64> %0, i32 7)
  store <2 x i64> %1, ptr @llvm_mips_bseti_d_RES
  ret void
}
declare <2 x i64> @llvm.mips.bseti.d(<2 x i64>, i32) nounwind
