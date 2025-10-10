	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.section	.text._Z8micro_tkILi128EEv13micro_globalsIXT_EE,"axG",@progbits,_Z8micro_tkILi128EEv13micro_globalsIXT_EE,comdat
	.protected	_Z8micro_tkILi128EEv13micro_globalsIXT_EE ; -- Begin function _Z8micro_tkILi128EEv13micro_globalsIXT_EE
	.globl	_Z8micro_tkILi128EEv13micro_globalsIXT_EE
	.p2align	8
	.type	_Z8micro_tkILi128EEv13micro_globalsIXT_EE,@function
_Z8micro_tkILi128EEv13micro_globalsIXT_EE: ; @_Z8micro_tkILi128EEv13micro_globalsIXT_EE
; %bb.0:                                ; %entry
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dwordx4 s[8:11], s[0:1], 0x18
	s_cmp_lg_u32 0, -1
	s_waitcnt lgkmcnt(0)
	s_cselect_b32 s9, 0, 0
	s_and_b32 s6, s9, -16
	v_lshlrev_b32_e32 v1, 4, v0
	s_mul_i32 s8, s8, s10
	s_mov_b32 s3, 0
	s_and_b32 s2, s9, 15
	s_add_i32 s11, s6, 16
	s_lshl_b32 s6, s8, 6
	v_and_b32_e32 v3, 0xc00, v1
	v_bitop3_b32 v1, v0, v1, 32 bitop3:0x6c
	v_lshrrev_b32_e32 v2, 1, v0
	s_cmp_eq_u64 s[2:3], 0
	v_lshrrev_b32_e32 v1, 1, v1
	v_and_b32_e32 v2, 0x60, v2
	v_bfe_u32 v4, v0, 2, 4
	v_and_or_b32 v2, v1, 24, v2
	s_cselect_b32 s12, s9, s11
	v_add_u32_e32 v1, s12, v3
	v_mad_u64_u32 v[2:3], s[2:3], v4, s8, v[2:3]
	v_readfirstlane_b32 s2, v1
	s_mov_b32 m0, s2
	s_lshl_b32 s2, s8, 4
	v_add_u32_e32 v1, 0x1000, v1
	s_mov_b32 s7, 0x110000
	v_lshlrev_b32_e32 v3, 1, v2
	v_add_lshl_u32 v2, v2, s2, 1
	v_readfirstlane_b32 s2, v1
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	buffer_load_dwordx4 v3, s[4:7], 0 offen lds
	s_mov_b32 m0, s2
	s_nop 0
	buffer_load_dwordx4 v2, s[4:7], 0 offen lds
	s_load_dwordx2 s[4:5], s[0:1], 0x30
	s_load_dwordx4 s[8:11], s[0:1], 0x48
	s_mov_b32 s6, -1
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_lshlrev_b32_e32 v2, 6, v0
	v_lshlrev_b32_e32 v3, 2, v0
	v_and_b32_e32 v1, 48, v0
	v_and_b32_e32 v2, 0x3c0, v2
	v_and_b32_e32 v3, 32, v3
	v_bitop3_b32 v2, v2, v3, v1 bitop3:0x36
	v_add_u32_e32 v3, s12, v2
	;;#ASMSTART
	ds_read_b128 a[0x70:0x73], v3 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x74:0x77], v3 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x78:0x7b], v3 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x7c:0x7f], v3 offset:0xc00
	;;#ASMEND
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_and_b32_e32 v0, 15, v0
	s_mul_i32 s0, s8, s10
	v_mul_lo_u32 v0, v0, s0
	s_mov_b32 s7, 0x20000
	v_lshl_add_u32 v0, v0, 1, v1
	;;#ASMSTART
	buffer_store_dwordx4 a[0x70:0x73], v0, s[4:7], 0 offen
	;;#ASMEND
	v_add_u32_e32 v1, 64, v0
	;;#ASMSTART
	buffer_store_dwordx4 a[0x74:0x77], v1, s[4:7], 0 offen
	;;#ASMEND
	v_add_u32_e32 v3, 0x80, v0
	;;#ASMSTART
	buffer_store_dwordx4 a[0x78:0x7b], v3, s[4:7], 0 offen
	;;#ASMEND
	v_add_u32_e32 v4, 0xc0, v0
	;;#ASMSTART
	buffer_store_dwordx4 a[0x7c:0x7f], v4, s[4:7], 0 offen
	;;#ASMEND
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_addk_i32 s12, 0x1000
	v_add_u32_e32 v2, s12, v2
	;;#ASMSTART
	ds_read_b128 a[0x70:0x73], v2 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x74:0x77], v2 offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x78:0x7b], v2 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 a[0x7c:0x7f], v2 offset:0xc00
	;;#ASMEND
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_lshl_b32 s0, s0, 4
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s4, s4, s0
	s_addc_u32 s5, s5, s1
	;;#ASMSTART
	buffer_store_dwordx4 a[0x70:0x73], v0, s[4:7], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_store_dwordx4 a[0x74:0x77], v1, s[4:7], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_store_dwordx4 a[0x78:0x7b], v3, s[4:7], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_store_dwordx4 a[0x7c:0x7f], v4, s[4:7], 0 offen
	;;#ASMEND
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z8micro_tkILi128EEv13micro_globalsIXT_EE
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 96
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 136
		.amdhsa_next_free_sgpr 13
		.amdhsa_accum_offset 8
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z8micro_tkILi128EEv13micro_globalsIXT_EE,"axG",@progbits,_Z8micro_tkILi128EEv13micro_globalsIXT_EE,comdat
.Lfunc_end0:
	.size	_Z8micro_tkILi128EEv13micro_globalsIXT_EE, .Lfunc_end0-_Z8micro_tkILi128EEv13micro_globalsIXT_EE
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 620
; NumSgprs: 19
; NumVgprs: 5
; NumAgprs: 128
; TotalNumVgprs: 136
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 16
; NumSGPRsForWavesPerEU: 19
; NumVGPRsForWavesPerEU: 136
; AccumOffset: 8
; Occupancy: 3
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 1
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_59756c0a8cc8bc03,@object ; @__hip_cuid_59756c0a8cc8bc03
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_59756c0a8cc8bc03
__hip_cuid_59756c0a8cc8bc03:
	.byte	0                               ; 0x0
	.size	__hip_cuid_59756c0a8cc8bc03, 1

	.ident	"AMD clang version 19.0.0git (ssh://github-emu/AMD-Lightning-Internal/llvm-project  25164 2b159522a6e9b34fe13b1d7b4c4ae751ef122765)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_59756c0a8cc8bc03
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     128
    .args:
      - .offset:         0
        .size:           96
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z8micro_tkILi128EEv13micro_globalsIXT_EE
    .private_segment_fixed_size: 0
    .sgpr_count:     19
    .sgpr_spill_count: 0
    .symbol:         _Z8micro_tkILi128EEv13micro_globalsIXT_EE.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     136
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
