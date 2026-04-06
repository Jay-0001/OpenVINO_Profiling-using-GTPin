
#ifdef intel_convert_as_bfloat16_float
#define _convert_as_bfloat16_float(val) intel_convert_as_bfloat16_float(val)
#else
inline float _convert_as_bfloat16_float(ushort source) {
 uint u = 0;
 if ( (source>>15) ) {
 u = 1 << 31;
 }
 u += ( ( (source >> 7) & 0b11111111)) << 23;
 u += (source & 0b1111111) << 16;
 float* f = (float*)&u;
 return *f;
}
#endif
#ifdef intel_convert_bfloat16_as_ushort
#define _convert_bfloat16_as_ushort(val) intel_convert_bfloat16_as_ushort(val)
#else
inline ushort _convert_bfloat16_as_ushort(float source) {
 uint* in = (uint*)&source;
 ushort u = 0;
 if ( (*in>>31) ) {
 u = 1 << 15;
 }
 u += ( ( (*in >> 23) & 0b11111111)) << 7;
 u += (*in >> 16) & 0b1111111;
 return u;
}
#endif

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#if !defined(cl_intel_subgroups) && defined(cl_khr_subgroups)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif
#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)
#define OFFSET_GLOBAL_PTR(elem_type, ptr, byte_offset) ((__global elem_type*)((__global char*)(ptr) + (byte_offset)))
#define MULTIPLY_OFFSET(elem_type, byte_offset) ((byte_offset) * sizeof(elem_type))
#if OPT_HINTS_SUPPORTED
# define ASSUME_HINT(x) __builtin_assume(x)
#else
# define ASSUME_HINT(x) do { } while (0)
#endif
#define unroll_for __attribute__((opencl_unroll_hint)) for
#define CEIL_DIV(a, b) (((a) + (b) - 1)/(b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(v,l,u) MAX((l),MIN((v),(u)))
#define MAKE_VECTOR_TYPE_IMPL_1(elem_type) elem_type
#define MAKE_VECTOR_TYPE_IMPL_2(elem_type) CAT(elem_type, 2)
#define MAKE_VECTOR_TYPE_IMPL_3(elem_type) CAT(elem_type, 3)
#define MAKE_VECTOR_TYPE_IMPL_4(elem_type) CAT(elem_type, 4)
#define MAKE_VECTOR_TYPE_IMPL_8(elem_type) CAT(elem_type, 8)
#define MAKE_VECTOR_TYPE_IMPL_16(elem_type) CAT(elem_type, 16)
#define MAKE_VECTOR_TYPE(elem_type, size) CAT(MAKE_VECTOR_TYPE_IMPL_, size)(elem_type)
#define AS_TYPE(type, val) CAT(as_, type)(val)
#define TYPE_SIZE_uchar 1
#define TYPE_SIZE_char 1
#define TYPE_SIZE_ushort 2
#define TYPE_SIZE_short 2
#define TYPE_SIZE_half 2
#define TYPE_SIZE_int 4
#define TYPE_SIZE_uint 4
#define TYPE_SIZE_float 4
#define TYPE_SIZE_ulong 8
#define TYPE_SIZE_long 8
#define TYPE_SIZE(type) CAT(TYPE_SIZE_, type)
#ifdef cl_intel_required_subgroup_size
#define REQD_SUB_GROUP_SIZE(sg_size) __attribute__((intel_reqd_sub_group_size(sg_size)))
#else
#define REQD_SUB_GROUP_SIZE(sg_size)
#endif

#define GET_DATA_INDEX(prefix, b, f, y, x) \
 CAT(prefix, _OFFSET) + \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (f)*CAT(prefix, _FEATURE_PITCH) + \
 (b)*CAT(prefix, _BATCH_PITCH)
#define GET_DATA_INDEX_RAW(prefix, i0, i1, i2, i3) \
 CAT(prefix, _OFFSET) + \
 (i0)*CAT(prefix, _PITCHES)[0] + \
 (i1)*CAT(prefix, _PITCHES)[1] + \
 (i2)*CAT(prefix, _PITCHES)[2] + \
 (i3)*CAT(prefix, _PITCHES)[3]
#define GET_DATA_INDEX_SAFE(prefix, b, f, y, x) \
 CAT(prefix, _OFFSET) + \
 (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) + \
 (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) + \
 (f % CAT(prefix, _FEATURE_NUM))*CAT(prefix, _FEATURE_PITCH) + \
 (b % CAT(prefix, _BATCH_NUM ))*CAT(prefix, _BATCH_PITCH)
 #define GET_DATA_INDEX_5D(prefix, b, f, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (z)*CAT(prefix, _Z_PITCH) + \
 (f)*CAT(prefix, _FEATURE_PITCH) + \
 (b)*CAT(prefix, _BATCH_PITCH)
#define GET_DATA_INDEX_5D_RAW(prefix, i0, i1, i2, i3, i4) \
 CAT(prefix, _OFFSET) + \
 (i0)*CAT(prefix, _PITCHES)[0] + \
 (i1)*CAT(prefix, _PITCHES)[1] + \
 (i2)*CAT(prefix, _PITCHES)[2] + \
 (i3)*CAT(prefix, _PITCHES)[3] + \
 (i4)*CAT(prefix, _PITCHES)[4]
#define GET_DATA_INDEX_5D_SAFE(prefix, b, f, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) + \
 (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) + \
 (z % CAT(prefix, _SIZE_Z ))*CAT(prefix, _Z_PITCH) + \
 (f % CAT(prefix, _FEATURE_NUM))*CAT(prefix, _FEATURE_PITCH) + \
 (b % CAT(prefix, _BATCH_NUM ))*CAT(prefix, _BATCH_PITCH)
#define GET_DATA_INDEX_6D(prefix, b, f, w, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (z)*CAT(prefix, _Z_PITCH) + \
 (w)*CAT(prefix, _W_PITCH) + \
 (f)*CAT(prefix, _FEATURE_PITCH) + \
 (b)*CAT(prefix, _BATCH_PITCH)
#define GET_DATA_INDEX_6D_SAFE(prefix, b, f, w, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) + \
 (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) + \
 (z % CAT(prefix, _SIZE_Z ))*CAT(prefix, _Z_PITCH) + \
 (w % CAT(prefix, _SIZE_W ))*CAT(prefix, _W_PITCH) + \
 (f % CAT(prefix, _FEATURE_NUM))*CAT(prefix, _FEATURE_PITCH) + \
 (b % CAT(prefix, _BATCH_NUM ))*CAT(prefix, _BATCH_PITCH)
#define GET_DATA_INDEX_6D_RAW(prefix, i0, i1, i2, i3, i4, i5) \
 CAT(prefix, _OFFSET) + \
 (i0)*CAT(prefix, _PITCHES)[0] + \
 (i1)*CAT(prefix, _PITCHES)[1] + \
 (i2)*CAT(prefix, _PITCHES)[2] + \
 (i3)*CAT(prefix, _PITCHES)[3] + \
 (i4)*CAT(prefix, _PITCHES)[4] + \
 (i5)*CAT(prefix, _PITCHES)[5]
#define GET_DATA_INDEX_7D(prefix, b, f, u, w, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (z)*CAT(prefix, _Z_PITCH) + \
 (w)*CAT(prefix, _W_PITCH) + \
 (u)*CAT(prefix, _U_PITCH) + \
 (f)*CAT(prefix, _FEATURE_PITCH) + \
 (b)*CAT(prefix, _BATCH_PITCH)
#define GET_DATA_INDEX_7D_SAFE(prefix, b, f, u, w, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) + \
 (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) + \
 (z % CAT(prefix, _SIZE_Z ))*CAT(prefix, _Z_PITCH) + \
 (w % CAT(prefix, _SIZE_W ))*CAT(prefix, _W_PITCH) + \
 (u % CAT(prefix, _SIZE_U ))*CAT(prefix, _U_PITCH) + \
 (f % CAT(prefix, _FEATURE_NUM))*CAT(prefix, _FEATURE_PITCH) + \
 (b % CAT(prefix, _BATCH_NUM ))*CAT(prefix, _BATCH_PITCH)
#define GET_DATA_INDEX_7D_RAW(prefix, i0, i1, i2, i3, i4, i5, i6) \
 CAT(prefix, _OFFSET) + \
 (i0)*CAT(prefix, _PITCHES)[0] + \
 (i1)*CAT(prefix, _PITCHES)[1] + \
 (i2)*CAT(prefix, _PITCHES)[2] + \
 (i3)*CAT(prefix, _PITCHES)[3] + \
 (i4)*CAT(prefix, _PITCHES)[4] + \
 (i5)*CAT(prefix, _PITCHES)[5] + \
 (i6)*CAT(prefix, _PITCHES)[6]
#define GET_DATA_INDEX_8D(prefix, b, f, v, u, w, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (z)*CAT(prefix, _Z_PITCH) + \
 (w)*CAT(prefix, _W_PITCH) + \
 (u)*CAT(prefix, _U_PITCH) + \
 (v)*CAT(prefix, _V_PITCH) + \
 (f)*CAT(prefix, _FEATURE_PITCH) + \
 (b)*CAT(prefix, _BATCH_PITCH)
#define GET_DATA_INDEX_8D_SAFE(prefix, b, f, v, u, w, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) + \
 (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) + \
 (z % CAT(prefix, _SIZE_Z ))*CAT(prefix, _Z_PITCH) + \
 (w % CAT(prefix, _SIZE_W ))*CAT(prefix, _W_PITCH) + \
 (u % CAT(prefix, _SIZE_U ))*CAT(prefix, _U_PITCH) + \
 (v % CAT(prefix, _SIZE_V ))*CAT(prefix, _V_PITCH) + \
 (f % CAT(prefix, _FEATURE_NUM))*CAT(prefix, _FEATURE_PITCH) + \
 (b % CAT(prefix, _BATCH_NUM ))*CAT(prefix, _BATCH_PITCH)
#define GET_DATA_INDEX_8D_RAW(prefix, i0, i1, i2, i3, i4, i5, i6, i7) \
 CAT(prefix, _OFFSET) + \
 (i0)*CAT(prefix, _PITCHES)[0] + \
 (i1)*CAT(prefix, _PITCHES)[1] + \
 (i2)*CAT(prefix, _PITCHES)[2] + \
 (i3)*CAT(prefix, _PITCHES)[3] + \
 (i4)*CAT(prefix, _PITCHES)[4] + \
 (i5)*CAT(prefix, _PITCHES)[5] + \
 (i6)*CAT(prefix, _PITCHES)[6] + \
 (i7)*CAT(prefix, _PITCHES)[7]
#define GET_DATA_BS_FYX_BSV8_INDEX(prefix, b, f, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((b) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (f)*CAT(prefix, _FEATURE_PITCH) + \
 ((b) / (sub_group_size))*CAT(prefix, _BATCH_PITCH) \
 )
inline uint get_b_fs_yx_fsv_index(uint b, uint f, uint y, uint x,
 uint x_size, uint y_size, uint f_size, uint b_size,
 uint b_pad_before, uint b_pad_after,
 uint f_pad_before, uint f_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after, uint alignment) {
 const uint feature = f + f_pad_before;
 const uint fs = feature / alignment;
 const uint fsv = feature % alignment;
 const uint x_pitch = alignment;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint fs_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);
 const uint output_offset = (b_pad_before + b) * b_pitch +
 fs * fs_pitch +
 (y_pad_before + y) * y_pitch +
 (x_pad_before + x) * x_pitch
 + fsv;
 return output_offset;
}
inline uint get_b_fs_yx_fsv_index_safe(uint b, uint f, uint y, uint x,
 uint x_size, uint y_size, uint f_size, uint b_size,
 uint b_pad_before, uint b_pad_after,
 uint f_pad_before, uint f_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after, uint alignment) {
 const uint f_mod = f_pad_before + (f % f_size);
 const uint fs = f_mod / alignment;
 const uint fsv = f_mod % alignment;
 const uint x_pitch = alignment;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint fs_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);
 const uint output_offset = (b_pad_before + (b % b_size)) * b_pitch +
 fs * fs_pitch +
 (y_pad_before + (y % y_size)) * y_pitch +
 (x_pad_before + (x % x_size)) * x_pitch
 + fsv;
 return output_offset;
}
#define GET_DATA_B_FS_YX_FSV16_INDEX(prefix, b, f, y, x) \
 get_b_fs_yx_fsv_index( \
 b, f, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16)
#define GET_DATA_B_FS_YX_FSV16_INDEX_SAFE(prefix, b, f, y, x) \
 get_b_fs_yx_fsv_index_safe( \
 b, f, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16)
#define GET_DATA_B_FS_YX_FSV2_INDEX(prefix, b, f, y, x) \
 get_b_fs_yx_fsv_index( \
 b, f, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 2)
#define GET_DATA_B_FS_YX_FSV2_INDEX_SAFE(prefix, b, f, y, x) \
 get_b_fs_yx_fsv_index_safe( \
 b, f, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 2)
#define GET_DATA_B_FS_YX_FSV4_INDEX(prefix, b, f, y, x) \
 get_b_fs_yx_fsv_index( \
 b, f, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 4)
#define GET_DATA_B_FS_YX_FSV4_INDEX_SAFE(prefix, b, f, y, x) \
 get_b_fs_yx_fsv_index_safe( \
 b, f, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 4)
#define GET_DATA_B_FS_YX_FSV8_INDEX(prefix, b, f, y, x) \
 get_b_fs_yx_fsv_index( \
 b, f, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 8)
#define GET_DATA_B_FS_YX_FSV8_INDEX_SAFE(prefix, b, f, y, x) \
 get_b_fs_yx_fsv_index_safe( \
 b, f, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 8)
#define GET_DATA_B_FS_YX_FSV32_INDEX(prefix, b, f, y, x) \
 get_b_fs_yx_fsv_index( \
 b, f, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 32)
#define GET_DATA_B_FS_YX_FSV32_INDEX_SAFE(prefix, b, f, y, x) \
 get_b_fs_yx_fsv_index_safe( \
 b, f, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 32)
inline uint get_fs_b_yx_fsv32_index(uint b, uint f, uint y, uint x,
 uint x_pad_before, uint x_size, uint x_pad_after,
 uint y_pad_before, uint y_size, uint y_pad_after,
 uint f_pad_before,
 uint size_b)
{
 const uint feature_tile_size = 32;
 const uint x_total_size = x_pad_before + x_size + x_pad_after;
 const uint y_total_size = y_pad_before + y_size + y_pad_after;
 const uint real_x = x + x_pad_before;
 const uint real_y = y + y_pad_before;
 const uint real_f = f + f_pad_before;
 const uint x_pitch = feature_tile_size;
 const uint y_pitch = x_pitch * x_total_size;
 const uint b_pitch = y_pitch * y_total_size;
 const uint f_tile_pitch = b_pitch * size_b;
 const uint feature_tile_number = real_f / feature_tile_size;
 const uint feature_local_number = real_f % feature_tile_size;
 size_t index = 0;
 index += feature_tile_number * f_tile_pitch;
 index += b * b_pitch;
 index += real_y * y_pitch;
 index += real_x * x_pitch;
 index += feature_local_number;
 return index;
}
inline uint get_fs_b_yx_fsv32_index_safe(uint b, uint f, uint y, uint x,
 uint x_pad_before, uint x_size, uint x_pad_after,
 uint y_pad_before, uint y_size, uint y_pad_after,
 uint f_pad_before, uint f_size,
 uint size_b)
{
 const uint feature_tile_size = 32;
 const uint x_total_size = x_pad_before + x_size + x_pad_after;
 const uint y_total_size = y_pad_before + y_size + y_pad_after;
 const uint real_x = (x % x_size) + x_pad_before;
 const uint real_y = (y % y_size) + y_pad_before;
 const uint real_f = (f % f_size) + f_pad_before;
 const uint x_pitch = feature_tile_size;
 const uint y_pitch = x_pitch * x_total_size;
 const uint b_pitch = y_pitch * y_total_size;
 const uint f_tile_pitch = b_pitch * size_b;
 const uint feature_tile_number = real_f / feature_tile_size;
 const uint feature_local_number = real_f % feature_tile_size;
 size_t index = 0;
 index += feature_tile_number * f_tile_pitch;
 index += b * b_pitch;
 index += real_y * y_pitch;
 index += real_x * x_pitch;
 index += feature_local_number;
 return index;
}
#define GET_DATA_FS_B_YX_FSV32_INDEX(prefix, b, f, y, x) \
 get_fs_b_yx_fsv32_index( \
 b, f, y, x, \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM))
#define GET_DATA_FS_B_YX_FSV32_INDEX_SAFE(prefix, b, f, y, x) \
 get_fs_b_yx_fsv32_index_safe( \
 b, f, y, x, \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM))
#define GET_DATA_B_FS_ZYX_FSV2_INDEX(prefix, b, f, z, y, x) \
 get_b_fs_zyx_fsv_index( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 2)
#define GET_DATA_B_FS_ZYX_FSV2_INDEX_SAFE(prefix, b, f, z, y, x) \
 get_b_fs_zyx_fsv_index_safe( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 2)
#define GET_DATA_B_FS_ZYX_FSV4_INDEX(prefix, b, f, z, y, x) \
 get_b_fs_zyx_fsv_index( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 4)
#define GET_DATA_B_FS_ZYX_FSV4_INDEX_SAFE(prefix, b, f, z, y, x) \
 get_b_fs_zyx_fsv_index_safe( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 4)
#define GET_DATA_B_FS_ZYX_FSV8_INDEX(prefix, b, f, z, y, x) \
 get_b_fs_zyx_fsv_index( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 8)
#define GET_DATA_B_FS_ZYX_FSV8_INDEX_SAFE(prefix, b, f, z, y, x) \
 get_b_fs_zyx_fsv_index_safe( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 8)
#define GET_DATA_B_FS_ZYX_FSV16_INDEX(prefix, b, f, z, y, x) \
 get_b_fs_zyx_fsv_index( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16)
#define GET_DATA_B_FS_ZYX_FSV16_INDEX_SAFE(prefix, b, f, z, y, x) \
 get_b_fs_zyx_fsv_index_safe( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16)
#define GET_DATA_B_FS_ZYX_FSV32_INDEX(prefix, b, f, z, y, x) \
 get_b_fs_zyx_fsv_index( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 32)
#define GET_DATA_B_FS_ZYX_FSV32_INDEX_SAFE(prefix, b, f, z, y, x) \
 get_b_fs_zyx_fsv_index_safe( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 32)
inline uint get_b_fs_zyx_fsv_index(uint b, uint f, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint f_size,
 uint b_pad_before, uint b_pad_after,
 uint f_pad_before, uint f_pad_after,
 uint z_pad_before, uint z_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after,
 uint alignment)
{
 const uint feature = f + f_pad_before;
 const uint fs = feature / alignment;
 const uint fsv = feature % alignment;
 const uint x_pitch = alignment;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint z_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint fs_pitch = z_pitch * (z_pad_before + z_size + z_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);
 const uint output_offset = (b_pad_before + b) * b_pitch +
 fs * fs_pitch +
 (z_pad_before + z) * z_pitch +
 (y_pad_before + y) * y_pitch +
 (x_pad_before + x) * x_pitch
 + fsv;
 return output_offset;
}
inline uint get_b_fs_zyx_fsv_index_safe(uint b, uint f, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint f_size,
 uint b_pad_before, uint b_pad_after,
 uint f_pad_before, uint f_pad_after,
 uint z_pad_before, uint z_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after,
 uint alignment) {
 const uint f_mod = f_pad_before + (f % f_size);
 const uint fs = f_mod / alignment;
 const uint fsv = f_mod % alignment;
 const uint x_pitch = alignment;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint z_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint fs_pitch = z_pitch * (z_pad_before + z_size + z_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);
 const uint output_offset = (b_pad_before + b) * b_pitch +
 fs * fs_pitch +
 (z_pad_before + (z % z_size)) * z_pitch +
 (y_pad_before + (y % y_size)) * y_pitch +
 (x_pad_before + (x % x_size)) * x_pitch
 + fsv;
 return output_offset;
}
inline uint get_bs_fs_zyx_bsv_fsv_index_safe(uint b, uint f, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint f_size, uint b_size,
 uint b_pad_before, uint b_pad_after,
 uint f_pad_before, uint f_pad_after,
 uint z_pad_before, uint z_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after, uint alignmentB, uint alignmentF) {
 const uint b_mod = b_pad_before + (b % b_size);
 const uint f_mod = f_pad_before + (f % f_size);
 const uint bs = b_mod / alignmentB;
 const uint bsv = b_mod % alignmentB;
 const uint fs = f_mod / alignmentF;
 const uint fsv = f_mod % alignmentF;
 const uint x_pitch = alignmentF * alignmentB;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint z_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint fs_pitch = z_pitch * (z_pad_before + z_size + z_pad_after);
 const uint bs_pitch = fs_pitch * ((total_f_size + alignmentF - 1) / alignmentF);
 const uint output_offset = bs * bs_pitch +
 fs * fs_pitch +
 (z_pad_before + (z % z_size)) * z_pitch +
 (y_pad_before + (y % y_size)) * y_pitch +
 (x_pad_before + (x % x_size)) * x_pitch +
 (bsv * alignmentF)
 + fsv;
 return output_offset;
}
inline uint get_bs_fs_zyx_bsv_fsv_index(uint b, uint f, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint f_size,
 uint b_pad_before, uint b_pad_after,
 uint f_pad_before, uint f_pad_after,
 uint z_pad_before, uint z_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after,
 uint b_alignment, uint f_alignment) {
 const uint feature = f + f_pad_before;
 const uint fs = feature / f_alignment;
 const uint fsv = feature % f_alignment;
 const uint bs = (b + b_pad_before) / b_alignment;
 const uint bsv = (b + b_pad_before) % b_alignment;
 const uint bsv_pitch = f_alignment;
 const uint x_pitch = bsv_pitch * b_alignment;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint z_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint fs_pitch = z_pitch * (z_pad_before + z_size + z_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint bs_pitch = fs_pitch * ((total_f_size + f_alignment - 1) / f_alignment);
 const uint output_offset = bs * bs_pitch +
 fs * fs_pitch +
 (z_pad_before + z) * z_pitch +
 (y_pad_before + y) * y_pitch +
 (x_pad_before + x) * x_pitch +
 bsv * bsv_pitch
 + fsv;
 return output_offset;
}
#define GET_DATA_BS_FS_YX_BSV16_FSV16_INDEX(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 16)
#define GET_DATA_BS_FS_YX_BSV16_FSV32_INDEX(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 32)
#define GET_DATA_BS_FS_ZYX_BSV32_FSV32_INDEX(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 32, 32)
#define GET_DATA_BS_FS_YX_BSV32_FSV32_INDEX(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 32, 32)
#define GET_DATA_BS_FS_YX_BSV4_FSV4_INDEX(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 4, 4)
#define GET_DATA_BS_FS_ZYX_BSV16_FSV4_INDEX(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 4)
#define GET_DATA_BS_FS_YX_BSV16_FSV4_INDEX(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 4)
#define GET_DATA_BS_FS_ZYX_BSV16_FSV8_INDEX(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 8)
#define GET_DATA_BS_FS_YX_BSV16_FSV8_INDEX(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 8)
#define GET_DATA_BS_FS_ZYX_BSV8_FSV4_INDEX(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 8, 4)
#define GET_DATA_BS_FS_YX_BSV8_FSV4_INDEX(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 8, 4)
#define GET_DATA_BS_FS_ZYX_BSV16_FSV2_INDEX(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 2)
#define GET_DATA_BS_FS_YX_BSV16_FSV2_INDEX(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 2)
#define GET_DATA_BS_FS_ZYX_BSV8_FSV2_INDEX(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 8, 2)
#define GET_DATA_BS_FS_YX_BSV8_FSV2_INDEX(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 8, 2)
#define GET_DATA_BS_FS_YX_BSV4_FSV2_INDEX(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 4, 2)
#define GET_DATA_BS_FS_ZYX_BSV32_FSV16_INDEX(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 32, 16)
#define GET_DATA_BS_FS_YX_BSV32_FSV16_INDEX(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 32, 16)
#define GET_DATA_BS_FS_ZYX_BSV16_FSV32_INDEX(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 32)
#define GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 16)
#define GET_DATA_BS_FS_YX_BSV16_FSV16_INDEX_SAFE(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 16)
#define GET_DATA_BS_FS_ZYX_BSV32_FSV32_INDEX_SAFE(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 32, 32)
#define GET_DATA_BS_FS_YX_BSV32_FSV32_INDEX_SAFE(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 32, 32)
#define GET_DATA_BS_FS_YX_BSV4_FSV4_INDEX_SAFE(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 4, 4)
#define GET_DATA_BS_FS_YX_BSV16_FSV4_INDEX_SAFE(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 4)
#define GET_DATA_BS_FS_ZYX_BSV16_FSV4_INDEX_SAFE(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 4)
#define GET_DATA_BS_FS_YX_BSV16_FSV8_INDEX_SAFE(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 8)
#define GET_DATA_BS_FS_ZYX_BSV16_FSV8_INDEX_SAFE(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 8)
#define GET_DATA_BS_FS_YX_BSV8_FSV4_INDEX_SAFE(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 8, 4)
#define GET_DATA_BS_FS_ZYX_BSV8_FSV4_INDEX_SAFE(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 8, 4)
#define GET_DATA_BS_FS_YX_BSV16_FSV2_INDEX_SAFE(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 2)
#define GET_DATA_BS_FS_ZYX_BSV16_FSV2_INDEX_SAFE(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 2)
#define GET_DATA_BS_FS_YX_BSV8_FSV2_INDEX_SAFE(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 8, 2)
#define GET_DATA_BS_FS_ZYX_BSV8_FSV2_INDEX_SAFE(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 8, 2)
#define GET_DATA_BS_FS_YX_BSV4_FSV2_INDEX_SAFE(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 4, 2)
#define GET_DATA_BS_FS_ZYX_BSV32_FSV16_INDEX_SAFE(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 32, 16)
#define GET_DATA_BS_FS_YX_BSV32_FSV16_INDEX_SAFE(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 32, 16)
#define GET_DATA_BS_FS_ZYX_BSV16_FSV32_INDEX_SAFE(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 32)
#define GET_DATA_BS_FS_YX_BSV16_FSV32_INDEX_SAFE(prefix, b, f, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 32)
#define GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX_SAFE(prefix, b, f, z, y, x) \
 get_bs_fs_zyx_bsv_fsv_index_safe( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 16)

#define GET_FILTER_OS_IS_YX_ISV_OSV_INDEX(prefix, o, i, y, x, osv, isv) \
 get_os_is_zyx_isv_osv_index( \
 o, i, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 1, \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 osv, \
 isv \
 )
#define GET_FILTER_IS_OS_YX_OSV_ISV_INDEX(prefix, o, i, y, x, osv, isv) \
 get_os_is_zyx_isv_osv_index( \
 i, o, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 1, \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM), \
 isv, \
 osv \
 )
#define GET_FILTER_IS_OS_YX_ISV_OSV_INDEX(prefix, o, i, y, x, osv, isv) \
 get_is_os_zyx_isv_osv_index( \
 o, i, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 1, \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 osv, \
 isv \
 )
#define GET_FILTER_OS_IS_ZYX_ISV_OSV_INDEX(prefix, o, i, z, y, x, osv, isv) \
 get_os_is_zyx_isv_osv_index( \
 o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 osv, \
 isv \
 )
#define GET_FILTER_IS_OS_ZYX_ISV16_OSV16_INDEX(prefix, o, i, z, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*(sub_group_size)*CAT(prefix, _X_PITCH) + \
 (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) + \
 (z)*(sub_group_size)*CAT(prefix, _Z_PITCH) + \
 ((i) % (sub_group_size)) + \
 ((o) / (sub_group_size))*(sub_group_size)*CAT(prefix, _OFM_PITCH) + \
 ((i) / (sub_group_size))*CAT(prefix, _IFM_PITCH) \
 )
#define GET_FILTER_IS_OS_YX_ISV16_OSV16_INDEX(prefix, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*(sub_group_size)*CAT(prefix, _X_PITCH) + \
 (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) + \
 ((i) % (sub_group_size)) + \
 ((o) / (sub_group_size))*(sub_group_size)*CAT(prefix, _OFM_PITCH) + \
 ((i) / (sub_group_size))*CAT(prefix, _IFM_PITCH) \
 )
#define GET_FILTER_OS_IS_YX_ISV8_OSV16_ISV2_INDEX(prefix, o, i, y, x, sub_group_size) \
 get_os_is_zyx_isv8_osv16_isv2_index( \
 0, o, i, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _GROUPS_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFFSET) \
 )
#define GET_FILTER_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX(prefix, o, i, z, y, x, sub_group_size) \
 get_os_is_zyx_isv8_osv16_isv2_index( \
 0, o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _GROUPS_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFFSET) \
 )
inline uint get_os_is_zyx_isv_osv_index(uint o, uint i, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint i_size, uint o_size, uint osv_size, uint isv_size)
{
 const uint isv = i % isv_size;
 const uint osv = o % osv_size;
 const uint is = i / isv_size;
 const uint os = o / osv_size;
 const uint x_pitch = osv_size * isv_size;
 const uint y_pitch = x_pitch * x_size;
 const uint z_pitch = y_pitch * y_size;
 const uint is_pitch = z_pitch * z_size;
 const uint os_pitch = is_pitch * ((i_size + isv_size - 1) / isv_size);
 const uint output_offset =
 osv +
 isv * osv_size +
 x * x_pitch +
 y * y_pitch +
 z * z_pitch +
 is * is_pitch +
 os * os_pitch;
 return output_offset;
}
inline uint get_is_os_zyx_isv_osv_index(uint o, uint i, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint i_size, uint o_size, uint osv_size, uint isv_size)
{
 const uint isv = i % isv_size;
 const uint osv = o % osv_size;
 const uint is = i / isv_size;
 const uint os = o / osv_size;
 const uint x_pitch = osv_size * isv_size;
 const uint y_pitch = x_pitch * x_size;
 const uint z_pitch = y_pitch * y_size;
 const uint os_pitch = z_pitch * z_size;
 const uint is_pitch = os_pitch * ((o_size + osv_size - 1) / osv_size);
 const uint output_offset =
 osv +
 isv * osv_size +
 x * x_pitch +
 y * y_pitch +
 z * z_pitch +
 os * os_pitch +
 is * is_pitch;
 return output_offset;
}
inline uint get_os_is_zyx_osv_isv_index(uint o, uint i, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint i_size, uint o_size, uint osv_size, uint isv_size)
{
 const uint isv = i % isv_size;
 const uint osv = o % osv_size;
 const uint is = i / isv_size;
 const uint os = o / osv_size;
 const uint x_pitch = osv_size * isv_size;
 const uint y_pitch = x_pitch * x_size;
 const uint z_pitch = y_pitch * y_size;
 const uint is_pitch = z_pitch * z_size;
 const uint os_pitch = is_pitch * ((i_size + isv_size - 1) / isv_size);
 const uint output_offset =
 isv +
 osv * isv_size +
 x * x_pitch +
 y * y_pitch +
 z * z_pitch +
 is * is_pitch +
 os * os_pitch;
 return output_offset;
}
inline uint get_g_os_is_zyx_osv_isv_index(uint g, uint o, uint i, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint i_size, uint o_size, uint osv_size, uint isv_size)
{
 const uint isv = i % isv_size;
 const uint osv = o % osv_size;
 const uint is = i / isv_size;
 const uint os = o / osv_size;
 const uint x_pitch = osv_size * isv_size;
 const uint y_pitch = x_pitch * x_size;
 const uint z_pitch = y_pitch * y_size;
 const uint is_pitch = z_pitch * z_size;
 const uint os_pitch = is_pitch * ((i_size + isv_size - 1) / isv_size);
 const uint g_pitch = os_pitch * ((o_size + osv_size - 1) / osv_size);
 const uint output_offset =
 isv +
 osv * isv_size +
 x * x_pitch +
 y * y_pitch +
 z * z_pitch +
 is * is_pitch +
 os * os_pitch +
 g * g_pitch;
 return output_offset;
}
#define GET_FILTER_G_OS_IS_ZYX_OSV16_ISV16_INDEX(prefix, g, o, i, z, y, x) \
 get_g_os_is_zyx_osv_isv_index( \
 g, o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 16, \
 16)
#define GET_FILTER_OS_IS_YX_OSV16_ISV16_INDEX(prefix, o, i, y, x) \
 get_os_is_zyx_osv_isv_index( \
 o, i, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 1, \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 16, \
 16)
#define GET_FILTER_OS_IS_ZYX_OSV16_ISV16_INDEX(prefix, o, i, z, y, x) \
 get_os_is_zyx_osv_isv_index( \
 o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 16, \
 16)
#define GET_FILTER_OS_IS_ZYX_OSV32_ISV16_INDEX(prefix, o, i, z, y, x) \
 get_os_is_zyx_osv_isv_index( \
 o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 32, \
 16)
#define GET_FILTER_OS_IS_ZYX_OSV64_ISV16_INDEX(prefix, o, i, z, y, x) \
 get_os_is_zyx_osv_isv_index( \
 o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 64, \
 16)
#define GET_FILTER_G_OS_IS_YX_ISV8_OSV16_ISV2_INDEX(prefix, g, o, i, y, x, sub_group_size) \
 get_os_is_zyx_isv8_osv16_isv2_index( \
 g, o, i, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _GROUPS_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFFSET) \
 )
#define GET_FILTER_G_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX(prefix, g, o, i, z, y, x, sub_group_size) \
 get_os_is_zyx_isv8_osv16_isv2_index( \
 g, o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _GROUPS_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFFSET) \
 )
inline uint get_os_is_zyx_isv8_osv16_isv2_index(uint g, uint o, uint i, uint z, uint y, uint x, uint x_size, uint y_size, uint z_size,
 uint g_size, uint o_size, uint i_size, uint offset)
{
 const uint group_offset = g * o_size * i_size * z_size * y_size * x_size;
 const uint xyz_offset = (x + y * x_size + z * x_size * y_size)* 8*16*2;
 const uint i2_val = i % 2;
 const uint i2_slice = i / 2;
 const uint i8_v = i2_slice % 8;
 const uint i8_s = i2_slice / 8;
 const uint i2_offset = i2_val;
 const uint o_offset = (o % 16)*2 + (o / 16) * 16 * i_size * x_size * y_size * z_size;
 const uint i8_offset = 8*16*2* x_size*y_size*z_size * i8_s + 16*2*i8_v;
 const size_t idx = offset + group_offset + xyz_offset + i2_offset + i8_offset + o_offset;
 return idx;
}
inline uint get_os_zyxi_osv16_index(uint o, uint i, uint z, uint y, uint x, uint i_size, uint o_size, uint x_size, uint y_size, uint z_size)
{
 const size_t idx = o%16 + (o / 16)*i_size*x_size*y_size*z_size*16 +
 16 *(i+ x*i_size + y*i_size*x_size + z*i_size*x_size*y_size);
 return idx;
}
#define GET_FILTER_OS_ZYXI_OSV16(prefix, o, i, z, y, x) \
 get_os_zyxi_osv16_index( \
 o, i, z, y, x, CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z))
#define GET_FILTER_GOIYX(prefix, g, o, i, y, x) \
 CAT(prefix, _OFFSET) + \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 (o)*CAT(prefix, _OFM_PITCH) + \
 (g)*CAT(prefix, _GROUPS_PITCH)
#define GET_FILTER_GIOYX(prefix, g, o, i, y, x) \
 CAT(prefix, _OFFSET) + \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 (o)*CAT(prefix, _OFM_PITCH) + \
 (g)*CAT(prefix, _GROUPS_PITCH)
#define GET_FILTER_GIOYX_SAFE(prefix, g, o, i, y, x) \
 CAT(prefix, _OFFSET) + \
 (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) + \
 (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) + \
 (i % CAT(prefix, _IFM_NUM))*CAT(prefix, _IFM_PITCH) + \
 (o % CAT(prefix, _OFM_NUM))*CAT(prefix, _OFM_PITCH) + \
 (g % CAT(prefix, _GROUPS_NUM))*CAT(prefix, _GROUPS_PITCH)
#define GET_FILTER_GOIYX_SAFE(prefix, g, o, i, y, x) \
 CAT(prefix, _OFFSET) + \
 (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) + \
 (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) + \
 (i % CAT(prefix, _IFM_NUM))*CAT(prefix, _IFM_PITCH) + \
 (o % CAT(prefix, _OFM_NUM))*CAT(prefix, _OFM_PITCH) + \
 (g % CAT(prefix, _GROUPS_NUM))*CAT(prefix, _GROUPS_PITCH)
#define GET_FILTER_INDEX(prefix, g, o, i, y, x) GET_FILTER_GOIYX(prefix, g, o, i, y, x)
#define GET_FILTER_INDEX_SAFE(prefix, g, o, i, y, x) GET_FILTER_GOIYX_SAFE(prefix, g, o, i, y, x)
#define GET_FILTER_GOIZYX(prefix, g, o, i, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (z)*CAT(prefix, _Z_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 (o)*CAT(prefix, _OFM_PITCH) + \
 (g)*CAT(prefix, _GROUPS_PITCH)
#define GET_FILTER_GOIZYX_SAFE(prefix, g, o, i, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) + \
 (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) + \
 (z % CAT(prefix, _SIZE_Z ))*CAT(prefix, _Z_PITCH) + \
 (i % CAT(prefix, _IFM_NUM))*CAT(prefix, _IFM_PITCH) + \
 (o % CAT(prefix, _OFM_NUM))*CAT(prefix, _OFM_PITCH) + \
 (g % CAT(prefix, _GROUPS_NUM))*CAT(prefix, _GROUPS_PITCH)
#define GET_FILTER_GIOZYX(prefix, g, o, i, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (z)*CAT(prefix, _Z_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 (o)*CAT(prefix, _OFM_PITCH) + \
 (g)*CAT(prefix, _GROUPS_PITCH)
#define GET_FILTER_GIOZYX_SAFE(prefix, g, o, i, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) + \
 (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) + \
 (z % CAT(prefix, _SIZE_Z ))*CAT(prefix, _Z_PITCH) + \
 (i % CAT(prefix, _IFM_NUM))*CAT(prefix, _IFM_PITCH) + \
 (o % CAT(prefix, _OFM_NUM))*CAT(prefix, _OFM_PITCH) + \
 (g % CAT(prefix, _GROUPS_NUM))*CAT(prefix, _GROUPS_PITCH)
#define GET_FILTER_INDEX_5D(prefix, g, o, i, z, y, x) GET_FILTER_GOIZYX(prefix, g, o, i, z, y, x)
#define GET_FILTER_INDEX_5D_SAFE(prefix, g, o, i, z, y, x) GET_FILTER_GOIZYX_SAFE(prefix, g, o, i, z, y, x)
#define GET_FILTER_OS_IYX_OSV_INDEX(prefix, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH) \
 )
#define GET_FILTER_OS_IYX_OSV_INDEX_INT4_PACKED(prefix, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 ((o) / (sub_group_size))*(CAT(prefix, _OFM_PITCH)/2) \
 )
#define GET_FILTER_OS_IS_YX_OSV_ISV_INDEX_INT4_PACKED(prefix, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 ((o) / (sub_group_size))*(CAT(prefix, _OFM_PITCH)/2) \
 )
#define GET_FILTER_OS_IYX_OSV_ROTATE_180_INDEX(prefix, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (CAT(prefix, _SIZE_X ) - x - 1)*CAT(prefix, _X_PITCH) + \
 (CAT(prefix, _SIZE_Y ) - y - 1)*CAT(prefix, _Y_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH) \
 )
inline uint get_gi_yxs_os_yxsv2_osv_index(uint g, uint o, uint i, uint y, uint x, uint x_size, uint g_pitch, uint i_pitch,
 uint y_pitch, uint x_pitch, uint offset, uint sub_group_size)
{
 const uint aligned_ofm_line = x_pitch;
 const uint ifm_height_pitch = (i_pitch/aligned_ofm_line);
 const uint dst_height = i*ifm_height_pitch + y*x_size + x;
 const uint base_filter_index = y*x_size + x;
 const uint aligned_height = dst_height & 0xfffffffe;
 const uint base_filter_odd = (base_filter_index & 0x1);
 uint slice_id = o / sub_group_size;
 uint id_in_slice = o % sub_group_size;
 uint slice_pitch = 2*sub_group_size;
 uint offset_in_slice = (int)(sub_group_size*base_filter_odd);
 const uint in_line = (slice_pitch*slice_id + offset_in_slice + id_in_slice);
 size_t idx = offset + aligned_height*aligned_ofm_line + in_line;
 idx += g * g_pitch;
 return idx;
}
#define GET_FILTER_I_YXS_OS_YXSV2_OSV_INDEX(prefix, o, i, y, x, sub_group_size) \
 get_gi_yxs_os_yxsv2_osv_index( \
 0, o, i, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _GROUPS_PITCH), \
 CAT(prefix, _IFM_PITCH), \
 CAT(prefix, _Y_PITCH), \
 CAT(prefix, _X_PITCH), \
 CAT(prefix, _OFFSET), \
 sub_group_size)
inline uint get_giy_xs_os_xsv2_osv_index(uint g, uint o, uint i, uint y, uint x, uint x_size, uint g_pitch,
 uint i_pitch, uint y_pitch, uint x_pitch, uint offset, uint sub_group_size)
{
 const uint aligned_ofm_line = x_pitch;
 const uint ifm_height_pitch = (i_pitch/aligned_ofm_line);
 const uint aligned_x_line = y_pitch / x_pitch;
 const uint dst_height = i*ifm_height_pitch + y*aligned_x_line + x;
 const uint base_filter_index = x;
 const uint aligned_height = dst_height & 0xfffffffe;
 const uint base_filter_odd = (base_filter_index & 0x1);
 uint slice_id = o / sub_group_size;
 uint id_in_slice = o % sub_group_size;
 uint slice_pitch = 2*sub_group_size;
 uint offset_in_slice = (int)(sub_group_size*base_filter_odd);
 const bool last_line_in_base_filter = (x == (x_size - 1));
 if (last_line_in_base_filter && base_filter_odd == 0)
 {
 const uint element_in_slice = 32;
 slice_id = o / element_in_slice;
 id_in_slice = o % element_in_slice;
 slice_pitch = 2*element_in_slice;
 offset_in_slice = 0;
 }
 const uint in_line = (slice_pitch*slice_id + offset_in_slice + id_in_slice);
 size_t idx = offset + aligned_height*aligned_ofm_line + in_line;
 idx += g * g_pitch;
 return idx;
}
#define GET_FILTER_IY_XS_OS_XSV2_OSV_INDEX(prefix, o, i, y, x, sub_group_size) \
 get_giy_xs_os_xsv2_osv_index( \
 0, o, i, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _GROUPS_PITCH), \
 CAT(prefix, _IFM_PITCH), \
 CAT(prefix, _Y_PITCH), \
 CAT(prefix, _X_PITCH), \
 CAT(prefix, _OFFSET), \
 sub_group_size)
inline uint get_is_os_zyx_isa8_osv8_isv2_index(uint o, uint i, uint z, uint y, uint x, uint size_x,
 uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
 const uint isv2_idx = i % 2;
 const uint osv_idx = o % 8;
 const uint isv1_idx = (i / 2) % 8;
 const uint is_idx = i / 16;
 const uint os_idx = o / 8;
 const uint of_8_aligned = ((size_ofm + 7) / 8);
 size_t idx = offset +
 isv2_idx +
 osv_idx * 2 +
 isv1_idx * 8 * 2 +
 x * 8 * 8 * 2 +
 y * size_x * 8 * 8 * 2 +
 z * size_y * size_x * 8 * 8 * 2 +
 os_idx * size_z * size_y * size_x * 8 * 8 * 2 +
 is_idx * of_8_aligned * size_z * size_y * size_x * 8 * 8 * 2;
 return idx;
}
inline uint get_g_os_is_zyx_isa_osv_isv_index(uint g, uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset,
 uint isa, uint osv, uint isv)
{
 const uint isv2_idx = i % isv;
 const uint osv_idx = o % osv;
 const uint isv1_idx = (i / isv) % isa;
 const uint is_idx = i / (isa * isv);
 const uint os_idx = o / osv;
 const uint if_aligned = ((size_ifm + (isa * isv) - 1) / (isa * isv));
 const uint of_aligned = ((size_ofm + (osv - 1)) / osv);
 size_t idx = offset +
 isv2_idx +
 osv_idx * isv +
 isv1_idx * osv * isv +
 x * isa * osv * isv +
 y * size_x * isa * osv * isv +
 z * size_y * size_x * isa * osv * isv +
 is_idx * size_z * size_y * size_x * isa * osv * isv +
 os_idx * if_aligned * size_z * size_y * size_x * isa * osv * isv +
 g * of_aligned * if_aligned * size_z * size_y * size_x * isa * osv * isv;
 return idx;
}
#define GET_FILTER_G_OS_IS_ZYX_ISA_OSV_ISV_INDEX(prefix, g, o, i, z, y, x, isa, osv, isv) \
 get_g_os_is_zyx_isa_osv_isv_index( \
 g, o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET), \
 isa, osv, isv)
inline uint get_g_os_is_yx_isa8_osv8_isv4_index(uint g, uint o, uint i, uint y, uint x, uint size_x,
 uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
 const uint isv2_idx = i % 4;
 const uint osv_idx = o % 8;
 const uint isv1_idx = (i / 4) % 8;
 const uint is_idx = i / 32;
 const uint os_idx = o / 8;
 const uint if_32_aligned = ((size_ifm + 31) / 32);
 const uint of_8_aligned = ((size_ofm + 7) / 8);
 size_t idx = offset +
 isv2_idx +
 osv_idx * 4 +
 isv1_idx * 8 * 4 +
 x * 8 * 8 * 4 +
 y * size_x * 8 * 8 * 4 +
 is_idx * size_y * size_x * 4 * 8 * 8 +
 os_idx * if_32_aligned * size_y * size_x * 4 * 8 * 8 +
 g * of_8_aligned * if_32_aligned * size_y * size_x * 4 * 8 * 8;
 return idx;
}
#define GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_INDEX(prefix, o, i, y, x) \
 get_g_os_is_yx_isa8_osv8_isv4_index( \
 0, o, i, y, x, CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET))
inline uint get_is_os_yx_isa8_osv8_isv2_index(uint o, uint i, uint y, uint x, uint size_x,
 uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
	const uint isv2_idx = i % 2;
	const uint osv_idx = o % 8;
	const uint isv1_idx = (i / 2) % 8;
	const uint is_idx = i / 16;
	const uint os_idx = o / 8;
 const uint of_8_aligned = ((size_ofm + 7) / 8);
	size_t idx = offset +
 isv2_idx +
 osv_idx * 2 +
 isv1_idx * 8 * 2 +
 x * 8 * 8 * 2 +
 y * size_x * 8 * 8 * 2 +
 os_idx * size_y * size_x * 2 * 8 * 8 +
 is_idx * of_8_aligned * size_y * size_x * 2 * 8 * 8;
 return idx;
}
inline uint get_is_os_yx_isa8_osv8_isv4_index(uint o, uint i, uint y, uint x, uint size_x,
 uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
	const uint isv2_idx = i % 4;
	const uint osv_idx = o % 8;
	const uint isv1_idx = (i / 4) % 8;
	const uint is_idx = i / 32;
	const uint os_idx = o / 8;
 const uint of_8_aligned = ((size_ofm + 7) / 8);
	size_t idx = offset +
 isv2_idx +
 osv_idx * 4 +
 isv1_idx * 8 * 4 +
 x * 8 * 8 * 4 +
 y * size_x * 8 * 8 * 4 +
 os_idx * size_y * size_x * 4 * 8 * 8 +
 is_idx * of_8_aligned * size_y * size_x * 4 * 8 * 8;
 return idx;
}
inline uint get_is_os_yx_osa8_isv16_osv4_index(uint o, uint i, uint y, uint x, uint size_x,
 uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
	const uint osv2_idx = o % 4;
	const uint isv_idx = i % 16;
	const uint osv1_idx = (o / 4) % 8;
	const uint os_idx = o / 32;
	const uint is_idx = i / 16;
 const uint of_32_aligned = ((size_ofm + 31) / 32);
	size_t idx = offset +
 osv2_idx +
 isv_idx * 4 +
 osv1_idx * 16 * 4 +
 x * 8 * 16 * 4 +
 y * size_x * 8 * 16 * 4 +
 os_idx * size_y * size_x * 4 * 16 * 8 +
 is_idx * of_32_aligned * size_y * size_x * 4 * 16 * 8;
 return idx;
}
inline uint get_os_is_zyx_isa8_osv8_isv4_index(uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z,
 uint size_ifm, uint size_ofm, uint offset)
{
 const uint ifm_slices = (size_ifm + 31)/32;
 const uint isv2_idx = i % 4;
 const uint osv_idx = o % 8;
 const uint isv1_idx = (i / 4) % 8;
 const uint is_idx = i / 32;
 const uint os_idx = o / 8;
 size_t idx = offset + isv2_idx + 4 * (osv_idx + 8 * isv1_idx);
 idx += x * 4 * 8 * 8;
 idx += y * size_x * 4 * 8 * 8;
 idx += z * size_y * size_x * 4 * 8 * 8;
 idx += is_idx * size_z * size_y * size_x * 4 * 8 * 8;
 idx += os_idx * ifm_slices * size_z * size_y * size_x * 4 * 8 * 8;
 return idx;
}
#define GET_FILTER_OS_IS_ZYX_ISA8_OSV8_ISV4_INDEX(prefix, o, i, z, y, x) \
 get_os_is_zyx_isa8_osv8_isv4_index( \
 o, i, z, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET))
inline uint get_os_is_yx_isa8_osv16_isv4_index(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
 const uint f_32_aligned = ((size_ifm + 31)/32) * 32;
 const uint isv2_idx = i % 4;
 const uint osv_idx = o % 16;
 const uint isv1_idx = (i / 4) % 8;
 const uint is_idx = i / 32;
 const uint os_idx = o / 16;
 size_t idx = offset + isv2_idx + 4 * (osv_idx + 16 * isv1_idx);
 idx += x * 4 * 8 * 16;
 idx += y * size_x * 4 * 8 * 16;
 idx += is_idx * size_y * size_x * 4 * 8 * 16;
 idx += os_idx * (f_32_aligned/32) * size_y * size_x * 4 * 8 * 16;
 return idx;
}
#define GET_FILTER_OS_IS_YX_ISA8_OSV16_ISV4_INDEX(prefix, o, i, y, x) \
 get_os_is_yx_isa8_osv16_isv4_index( \
 o, i, y, x, CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET))
inline uint get_os_is_zyx_isa8_osv16_isv4_index(uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z,
 uint size_ifm, uint size_ofm, uint offset)
{
 const uint ifm_slices = (size_ifm + 31)/32;
 const uint isv2_idx = i % 4;
 const uint osv_idx = o % 16;
 const uint isv1_idx = (i / 4) % 8;
 const uint is_idx = i / 32;
 const uint os_idx = o / 16;
 size_t idx = offset + isv2_idx + 4 * (osv_idx + 16 * isv1_idx);
 idx += x * 4 * 8 * 16;
 idx += y * size_x * 4 * 8 * 16;
 idx += z * size_y * size_x * 4 * 8 * 16;
 idx += is_idx * size_z * size_y * size_x * 4 * 8 * 16;
 idx += os_idx * ifm_slices * size_z * size_y * size_x * 4 * 8 * 16;
 return idx;
}
#define GET_FILTER_OS_IS_ZYX_ISA8_OSV16_ISV4_INDEX(prefix, o, i, z, y, x) \
 get_os_is_zyx_isa8_osv16_isv4_index( \
 o, i, z, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET))
inline uint get_os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4_index(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
 const uint o_swizzled = (o % 4) * 8 + ((o % 32) / 4) + (o / 32) * 32;
 const uint isv_idx = i % 4;
 const uint isa_idx = (i / 4) % 8;
 const uint is_idx = (i / 32);
 const uint osv_idx = o_swizzled % 8;
 const uint osa_idx = (o_swizzled / 8) % 4;
 const uint os_idx = (o / 32);
 const uint f_32_aligned = ((size_ifm + 31)/32);
 size_t idx = offset +
 isv_idx +
 osv_idx * 4 +
 isa_idx * 8 * 4 +
 osa_idx * 8 * 32 +
 x * 32 * 32 +
 y * size_x * 32 * 32 +
 is_idx * 32 * 32 * size_x * size_y +
 os_idx * 32 * 32 * f_32_aligned * size_x * size_y;
 return idx;
}
inline uint get_os_is_yx_osa2_isa8_osv16_isv4_swizzled_by_2_index(
 uint o, uint i, uint y, uint x,
 uint size_x, uint size_y,
 uint size_ifm, uint size_ofm,
 uint offset)
{
 const uint o_swizzled = (o % 2) * 16 + ((o % 32) / 2) + (o / 32) * 32;
 const uint isv_idx = i % 4;
 const uint isa_idx = (i / 4) % 8;
 const uint is_idx = i / 32;
 const uint osv_idx = o_swizzled % 16;
 const uint osa_idx = (o_swizzled / 16) % 2;
 const uint os_idx = o / 32;
 const uint f_32_aligned = (size_ifm + 31) / 32;
 size_t idx = offset +
 isv_idx +
 osv_idx * 4 +
 isa_idx * 16 * 4 +
 osa_idx * 16 * 32 +
 x * 32 * 32 +
 y * size_x * 32 * 32 +
 is_idx * 32 * 32 * size_x * size_y +
 os_idx * 32 * 32 * f_32_aligned * size_x * size_y;
 return idx;
}
inline uint get_os_is_zyx_osa2_isa8_osv16_isv4_swizzled_by_2_index(
 uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z,
 uint size_ifm, uint size_ofm,
 uint offset)
{
 const uint o_swizzled = (o % 2) * 16 + ((o % 32) / 2) + (o / 32) * 32;
 const uint isv_idx = i % 4;
 const uint isa_idx = (i / 4) % 8;
 const uint is_idx = i / 32;
 const uint osv_idx = o_swizzled % 16;
 const uint osa_idx = (o_swizzled / 16) % 2;
 const uint os_idx = o / 32;
 const uint f_32_aligned = (size_ifm + 31) / 32;
 size_t idx = offset +
 isv_idx +
 osv_idx * 4 +
 isa_idx * 16 * 4 +
 osa_idx * 16 * 32 +
 x * 32 * 32 +
 y * size_x * 32 * 32 +
 z * size_x * size_y * 32 * 32 +
 is_idx * 32 * 32 * size_x * size_y * size_z +
 os_idx * 32 * 32 * f_32_aligned * size_x * size_y * size_z;
 return idx;
}
inline uint get_os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4_index(uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z,
 uint size_ifm, uint size_ofm, uint offset)
{
 const uint o_swizzled = (o % 4) * 8 + ((o % 32) / 4) + (o / 32) * 32;
 const uint isv_idx = i % 4;
 const uint isa_idx = (i / 4) % 8;
 const uint is_idx = (i / 32);
 const uint osv_idx = o_swizzled % 8;
 const uint osa_idx = (o_swizzled / 8) % 4;
 const uint os_idx = (o / 32);
 const uint f_32_aligned = ((size_ifm + 31)/32);
 size_t idx = offset +
 isv_idx +
 osv_idx * 4 +
 isa_idx * 8 * 4 +
 osa_idx * 8 * 32 +
 x * 32 * 32 +
 y * size_x * 32 * 32 +
 z * size_x * size_y * 32 * 32 +
 is_idx * 32 * 32 * size_x * size_y * size_z +
 os_idx * 32 * 32 * f_32_aligned * size_x * size_y * size_z;
 return idx;
}
inline uint get_g_is_os_yx_osa4_isa8_osv8_isv4(uint g, uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
 const uint isv_idx = i % 4;
 const uint isa_idx = (i / 4) % 8;
 const uint is_idx = (i / 32);
 const uint osv_idx = o % 8;
 const uint osa_idx = (o / 8) % 4;
 const uint os_idx = (o / 32);
 const uint ifm_32_aligned = ((size_ifm + 31) / 32);
 const uint ofm_32_aligned = ((size_ofm + 31) / 32);
 size_t idx = offset +
 isv_idx +
 osv_idx * 4 +
 isa_idx * 8 * 4 +
 osa_idx * 8 * 32 +
 x * 32 * 32 +
 y * size_x * 32 * 32 +
 z * size_y * size_x * 32 * 32 +
 os_idx * 32 * 32 * size_x * size_y * size_z +
 is_idx * 32 * 32 * ofm_32_aligned * size_x * size_y * size_z +
 g * 32 * 32 * ifm_32_aligned * ofm_32_aligned * size_x * size_y * size_z;
 return idx;
}
inline uint get_g_os_is_yx_osa4_isa8_osv8_isv4(uint g, uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
 const uint isv_idx = i % 4;
 const uint isa_idx = (i / 4) % 8;
 const uint is_idx = (i / 32);
 const uint osv_idx = o % 8;
 const uint osa_idx = (o / 8) % 4;
 const uint os_idx = (o / 32);
 const uint ifm_32_aligned = ((size_ifm + 31)/32);
 const uint ofm_32_aligned = ((size_ofm + 31)/32);
 size_t idx = offset +
 isv_idx +
 osv_idx * 4 +
 isa_idx * 8 * 4 +
 osa_idx * 8 * 32 +
 x * 32 * 32 +
 y * size_x * 32 * 32 +
 z * size_y * size_x * 32 * 32 +
 is_idx * 32 * 32 * size_x * size_y * size_z +
 os_idx * 32 * 32 * ifm_32_aligned * size_x * size_y * size_z +
 g * 32 * 32 * ifm_32_aligned * ofm_32_aligned * size_x * size_y * size_z;
 return idx;
}
inline uint get_g_os_is_yx_osa4_isa8_osv8_isv2(uint g, uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
 const uint isv_idx = i % 2;
 const uint isa_idx = (i / 2) % 8;
 const uint is_idx = (i / 16);
 const uint osv_idx = o % 8;
 const uint osa_idx = (o / 8) % 4;
 const uint os_idx = (o / 32);
 const uint ifm_16_aligned = ((size_ifm + 15)/16);
 const uint ofm_32_aligned = ((size_ofm + 31)/32);
 size_t idx = offset +
 isv_idx +
 osv_idx * 2 +
 isa_idx * 8 * 2 +
 osa_idx * 8 * 16 +
 x * 32 * 16 +
 y * size_x * 32 * 16 +
 z * size_y * size_x * 32 * 16 +
 is_idx * 32 * 16 * size_x * size_y * size_z +
 os_idx * 32 * 16 * ifm_16_aligned * size_x * size_y * size_z +
 g * 32 * 16 * ifm_16_aligned * ofm_32_aligned * size_x * size_y * size_z;
 return idx;
}
inline uint get_g_os_is_yx_osa2_isa8_osv8_isv2(uint g, uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
 const uint isv_idx = i % 2;
 const uint isa_idx = (i / 2) % 8;
 const uint is_idx = (i / 16);
 const uint osv_idx = o % 8;
 const uint osa_idx = (o / 8) % 2;
 const uint os_idx = (o / 16);
 const uint ifm_16_aligned = ((size_ifm + 15)/16);
 const uint ofm_16_aligned = ((size_ofm + 15)/16);
 size_t idx = offset +
 isv_idx +
 osv_idx * 2 +
 isa_idx * 8 * 2 +
 osa_idx * 8 * 16 +
 x * 16 * 16 +
 y * size_x * 16 * 16 +
 z * size_y * size_x * 16 * 16 +
 is_idx * 16 * 16 * size_x * size_y * size_z +
 os_idx * 16 * 16 * ifm_16_aligned * size_x * size_y * size_z +
 g * 16 * 16 * ifm_16_aligned * ofm_16_aligned * size_x * size_y * size_z;
 return idx;
}
inline uint get_g_is_os_yx_isa2_osa8_isv8_osv2(uint g, uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
 return get_g_os_is_yx_osa2_isa8_osv8_isv2(g, i, o, z, y, x, size_x, size_y, size_z, size_ofm, size_ifm, offset);
}
inline uint get_g_is_os_yx_isa4_osa8_isv8_osv4(uint g, uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z, uint size_ifm, uint size_ofm, uint offset)
{
 return get_g_os_is_yx_osa4_isa8_osv8_isv4(g, i, o, z, y, x, size_x, size_y, size_z, size_ofm, size_ifm, offset);
}
#define GET_FILTER_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_INDEX(prefix, o, i, y, x) \
 get_g_os_is_yx_osa4_isa8_osv8_isv4( \
 0, o, i, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 1, \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET))
#define GET_FILTER_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_INDEX(prefix, o, i, z, y, x) \
 get_g_os_is_yx_osa4_isa8_osv8_isv4( \
 0, o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET))
#define GET_FILTER_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(prefix, o, i, y, x) \
 get_os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4_index( \
 o, i, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET))
#define GET_FILTER_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(prefix, o, i, z, y, x) \
 get_os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4_index( \
 o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET))
#define GET_FILTER_OS_IS_YX_OSA2_ISA8_OSV16_ISV4_SWIZZLED_BY_2_INDEX(prefix, o, i, y, x) \
 get_os_is_yx_osa2_isa8_osv16_isv4_swizzled_by_2_index( \
 o, i, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET))
#define GET_FILTER_OS_IS_ZYX_OSA2_ISA8_OSV16_ISV4_SWIZZLED_BY_2_INDEX(prefix, o, i, z, y, x) \
 get_os_is_zyx_osa2_isa8_osv16_isv4_swizzled_by_2_index( \
 o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET))
inline uint get_is_o32_yx_isv32_swizzled_by_4_index(uint o, uint i, uint y, uint x, uint i_size, uint o_size, uint x_size, uint y_size)
{
 const uint o_aligned_to_32 = ((o_size + 31) / 32) * 32;
 const uint o_swizzled = (o % 4) * 8 + ((o % 32) / 4) + (o / 32) * 32;
 const uint i_aligned_to_32 = ((i_size + 31) / 32) * 32;
 const uint i_val = i % 32;
 const uint i_slice = i / 32;
 const size_t idx = i_val + 32* (x + x_size * (y + y_size * (o_swizzled + o_aligned_to_32 * i_slice) ) );
 return idx;
}
#define GET_FILTER_G_OS_IS_YX_OSV16_ISV4_INDEX(prefix, g, o, i, y, x) \
 get_g_os_is_yx_osv_isv( \
 g, o, i, y, x, \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), 16, 4)
inline uint get_g_os_is_yx_osv_isv(uint g, uint o, uint i, uint y, uint x,
 uint i_size,
 uint o_size,
 uint x_size,
 uint y_size,
 uint osv_size,
 uint isv_size)
{
 return get_g_os_is_zyx_osv_isv_index(g, o, i, 0, y, x,
 x_size, y_size, 1, i_size, o_size, osv_size, isv_size);
}
#define GET_FILTER_OS_IS_YX_OSV8_ISV4_INDEX(prefix, o, i, y, x) \
 get_g_os_is_yx_osv_isv( \
 0, o, i, y, x, \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), 8, 4)
#define GET_FILTER_OS_IS_YX_OSV16_ISV4_INDEX(prefix, o, i, y, x) \
 get_g_os_is_yx_osv_isv( \
 0, o, i, y, x, \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), 16, 4)
#define GET_FILTER_OS_IS_YX_OSV32_ISV4_INDEX(prefix, o, i, y, x) \
 get_g_os_is_yx_osv_isv( \
 0, o, i, y, x, \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), 32, 4)
#define GET_FILTER_OS_IS_ZYX_OSV32_ISV4_INDEX(prefix, o, i, z, y, x) \
 get_os_is_zyx_osv_isv_index( \
 o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 32, \
 4)
#define GET_FILTER_OS_IS_YX_OSV32_ISV4_SWIZZLED_BY_2_INDEX(prefix, o, i, y, x) \
 get_os_is_yx_osv32_isv4_swizzled_by_2( \
 o, i, y, x, \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_X))
inline uint get_os_is_yx_osv32_isv4_swizzled_by_2(uint o, uint i, uint y, uint x,
 uint o_size,
 uint i_size,
 uint y_size,
 uint x_size)
{
 const uint osv = 32;
 const uint os = o / osv;
 const uint ofm_block = (o % osv) % 2;
 const uint ofm_in_block = (o % osv) / 2;
 const uint tile = 4;
 const uint ifm_aligned = ((i_size + tile - 1) / tile) * tile;
 const uint ifm_tile = i / tile;
 const uint id = i - ifm_tile * tile;
 uint idx = os * ifm_aligned * y_size * x_size * osv
 + ifm_tile * y_size * x_size * osv * tile
 + y * x_size * osv * tile
 + x * osv * tile
 + ofm_block * 16 * tile
 + ofm_in_block * tile
 + id;
 return idx;
}
inline uint get_os_is_osv32_isv32_swizzled_by_4_index(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
 const uint size_ifm_a = ((size_ifm + 31)/32) * 32;
 const uint o_hi = o / 32;
 const uint o_lo = o % 32;
 const uint i_hi = i / 32;
 const uint i_lo = i % 32;
 const uint o_lo1 = o_lo % 4;
 const uint o_lo2 = (o_lo / 4) % 8;
 const uint i_lo1 = i_lo % 4;
 const uint i_lo2 = i_lo / 4;
 const uint idx_in_group = o_lo2 * 4 + o_lo1 * (32 * 8) + i_lo2 * 32 + i_lo1;
 const uint group_idx = o_hi * (size_ifm_a / 32) + i_hi;
 return group_idx * (32 * 32) + idx_in_group;
}
inline uint get_os_i_yxs_osv_yxsv4_index(uint o, uint i, uint y, uint x, uint i_size, uint size_x, uint size_y, uint osv) {
 const uint yxsv = 4;
 uint yx = y * size_x + x;
 uint yx_size_aligned = (size_x * size_y + yxsv - 1) / yxsv * yxsv;
 uint os_index = o / osv;
 uint yxs_index = yx / yxsv;
 uint osv_index = o % osv;
 uint yxsv_index = yx % yxsv;
 uint index = 0;
 index += yxsv_index;
 index += osv_index * yxsv;
 index += yxs_index * yxsv * osv;
 index += i * osv * yx_size_aligned;
 index += os_index * osv * yx_size_aligned * i_size;
 return index;
}
#define GET_FILTER_G_OS_IYX_OSV16(prefix, g, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 (g * CAT(prefix, _GROUPS_PITCH)) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH) \
 )
#define GET_FILTER_OS_IYX_OSV16(prefix, o, i, y, x, sub_group_size) GET_FILTER_G_OS_IYX_OSV16(prefix, 0, o, i, y, x, sub_group_size)
#define GET_FILTER_GS_OIYX_GSV16(prefix, g, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((g) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 (o)*CAT(prefix, _OFM_PITCH) + \
 ((g) / (sub_group_size))*CAT(prefix, _GROUPS_PITCH) \
 )
#define GET_FILTER_GS_OIZYX_GSV16(prefix, g, o, i, z, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((g) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (z)*CAT(prefix, _Z_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 (o)*CAT(prefix, _OFM_PITCH) + \
 ((g) / (sub_group_size))*CAT(prefix, _GROUPS_PITCH) \
 )
#define GET_FILTER_G_OS_IYX_OSV16_ROTATE_180(prefix, g, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 (g * CAT(prefix, _GROUPS_PITCH)) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (CAT(prefix, _SIZE_X ) - x - 1)*CAT(prefix, _X_PITCH) + \
 (CAT(prefix, _SIZE_Y ) - y - 1)*CAT(prefix, _Y_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH) \
 )
#define GET_FILTER_G_IS_OS_ZYX_ISV16_OSV16_INDEX(prefix, g, o, i, z, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 (g)*CAT(prefix, _GROUPS_PITCH) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*(sub_group_size)*CAT(prefix, _X_PITCH) + \
 (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) + \
 (z)*(sub_group_size)*CAT(prefix, _Z_PITCH) + \
 ((i) % (sub_group_size)) + \
 ((o) / (sub_group_size))*(sub_group_size)*CAT(prefix, _OFM_PITCH) + \
 ((i) / (sub_group_size))*CAT(prefix, _IFM_PITCH) \
 )
#define GET_FILTER_G_IS_OS_YX_ISV16_OSV16_INDEX(prefix, g, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 (g)*CAT(prefix, _GROUPS_PITCH) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*(sub_group_size)*CAT(prefix, _X_PITCH) + \
 (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) + \
 ((i) % (sub_group_size)) + \
 ((o) / (sub_group_size))*(sub_group_size)*CAT(prefix, _OFM_PITCH) + \
 ((i) / (sub_group_size))*CAT(prefix, _IFM_PITCH) \
 )
#define GET_FILTER_G_OS_IS_ZYX_ISV16_OSV16_INDEX(prefix, g, o, i, z, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 (g)*CAT(prefix, _GROUPS_PITCH) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*(sub_group_size)*CAT(prefix, _X_PITCH) + \
 (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) + \
 (z)*(sub_group_size)*CAT(prefix, _Z_PITCH) + \
 ((i) % (sub_group_size)) + \
 ((i) / (sub_group_size))*(sub_group_size)*CAT(prefix, _IFM_PITCH) + \
 ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH) \
 )
#define GET_FILTER_GI_YXS_OS_YXSV2_OSV_INDEX(prefix, g, o, i, y, x, sub_group_size) \
 get_gi_yxs_os_yxsv2_osv_index( \
 g, o, i, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _GROUPS_PITCH), \
 CAT(prefix, _IFM_PITCH), \
 CAT(prefix, _Y_PITCH), \
 CAT(prefix, _X_PITCH), \
 CAT(prefix, _OFFSET), \
 sub_group_size)
#define GET_FILTER_GIY_XS_OS_XSV2_OSV_INDEX(prefix, g, o, i, y, x, sub_group_size) \
 get_giy_xs_os_xsv2_osv_index( \
 g, o, i, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _GROUPS_PITCH), \
 CAT(prefix, _IFM_PITCH), \
 CAT(prefix, _Y_PITCH), \
 CAT(prefix, _X_PITCH), \
 CAT(prefix, _OFFSET), \
 sub_group_size)
inline uint get_gs_oi_yxs_gsv_yxsv4_index(uint g, uint o, uint i, uint y, uint x, uint o_size, uint i_size, uint size_x, uint size_y, const uint gsv) {
 const uint yxsv = 4;
 uint yx = y * size_x + x;
 uint yx_size_aligned = (size_x * size_y + yxsv - 1) / yxsv * yxsv;
 uint gs_index = g / gsv;
 uint yxs_index = yx / yxsv;
 uint gsv_index = g % gsv;
 uint yxsv_index = yx % yxsv;
 uint index = 0;
 index += yxsv_index;
 index += gsv_index * yxsv;
 index += yxs_index * yxsv * gsv;
 index += o * i * gsv * yx_size_aligned;
 index += gs_index * gsv * yx_size_aligned * o_size * i_size;
 return index;
}
#define GET_FILTER_GS_OI_YXS_GSV4_YXSV4_INDEX(prefix, g, o, i, y, x) \
 get_gs_oi_yxs_gsv_yxsv4_index( \
 g, o, i, y, x, \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 4)
#define GET_FILTER_GS_OI_YXS_GSV16_YXSV4_INDEX(prefix, g, o, i, y, x) \
 get_gs_oi_yxs_gsv_yxsv4_index( \
 g, o, i, y, x, \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 16)
#define GET_FILTER_GS_OI_YXS_GSV32_YXSV4_INDEX(prefix, g, o, i, y, x) \
 get_gs_oi_yxs_gsv_yxsv4_index( \
 g, o, i, y, x, \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 32)
#define GET_FILTER_G_OS_IS_YX_ISV16_OSV16_INDEX(prefix, g, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 (g * CAT(prefix, _GROUPS_PITCH)) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*(sub_group_size)*CAT(prefix, _X_PITCH) + \
 (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) + \
 ((i) % (sub_group_size)) + \
 ((i) / (sub_group_size))*(sub_group_size)*CAT(prefix, _IFM_PITCH) + \
 ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH) \
 )
inline uint get_g_os_zyx_is_osv_isv_index(uint g, uint o, uint i, uint z, uint y, uint x,
 uint g_size, uint o_size, uint i_size, uint z_size, uint y_size, uint x_size,
 uint osv, uint isv) {
 uint is_size = (i_size + isv - 1) / isv;
 uint os_size = (o_size + osv - 1) / osv;
 uint isv_index = i % isv;
 uint osv_index = o % osv;
 uint is_index = i / isv;
 uint os_index = o / osv;
 uint isv_pitch = 1;
 uint osv_pitch = isv_pitch * isv;
 uint is_pitch = osv_pitch * osv;
 uint x_pitch = is_pitch * is_size;
 uint y_pitch = x_pitch * x_size;
 uint z_pitch = y_pitch * y_size;
 uint os_pitch = z_pitch * z_size;
 uint g_pitch = os_pitch * os_size;
 uint index = 0;
 index += isv_index * isv_pitch;
 index += osv_index * osv_pitch;
 index += is_index * is_pitch;
 index += x * x_pitch;
 index += y * y_pitch;
 index += z * z_pitch;
 index += os_index * os_pitch;
 index += g * g_pitch;
 return index;
}
#define GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, osv, isv) \
 get_g_os_zyx_is_osv_isv_index( \
 g, o, i, z, y, x, \
 CAT(tensor, _GROUPS_NUM), \
 CAT(tensor, _OFM_NUM), \
 CAT(tensor, _IFM_NUM), \
 CAT(tensor, _SIZE_Z), \
 CAT(tensor, _SIZE_Y), \
 CAT(tensor, _SIZE_X), \
 osv, isv)
#define GET_FILTER_G_OS_ZYX_IS_OSV16_ISV4_INDEX(tensor, g, o, i, z, y, x) GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 16, 4)
#define GET_FILTER_G_OS_ZYX_IS_OSV16_ISV16_INDEX(tensor, g, o, i, z, y, x) GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 16, 16)
#define GET_FILTER_G_OS_ZYX_IS_OSV16_ISV32_INDEX(tensor, g, o, i, z, y, x) GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 16, 32)
#define GET_FILTER_G_OS_ZYX_IS_OSV32_ISV4_INDEX(tensor, g, o, i, z, y, x) GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 32, 4)
#define GET_FILTER_G_OS_ZYX_IS_OSV32_ISV16_INDEX(tensor, g, o, i, z, y, x) GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 32, 16)
#define GET_FILTER_G_OS_ZYX_IS_OSV32_ISV32_INDEX(tensor, g, o, i, z, y, x) GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 32, 32)
inline uint get_g_os_y_is_x_osv_isv_index(uint g, uint o, uint i, uint y, uint x,
 uint x_size, uint y_size, uint i_size, uint o_size, uint osv_size, uint isv_size)
{
 const uint isv = i % isv_size;
 const uint osv = o % osv_size;
 const uint is = i / isv_size;
 const uint os = o / osv_size;
 const uint x_pitch = osv_size * isv_size;
 const uint is_pitch = x_pitch * x_size;
 const uint y_pitch = is_pitch * ((i_size + isv_size - 1) / isv_size);
 const uint os_pitch = y_pitch * y_size;
 const uint g_pitch = os_pitch * ((o_size + osv_size - 1) / osv_size);
 const uint output_offset =
 isv +
 osv * isv_size +
 x * x_pitch +
 is * is_pitch +
 y * y_pitch +
 os * os_pitch +
 g * g_pitch;
 return output_offset;
}
#define GET_FILTER_G_OS_Y_IS_X_OSV_ISV_INDEX(tensor, g, o, i, y, x, osv, isv) \
 get_g_os_y_is_x_osv_isv_index( \
 g, o, i, y, x, \
 CAT(tensor, _SIZE_X), \
 CAT(tensor, _SIZE_Y), \
 CAT(tensor, _IFM_NUM), \
 CAT(tensor, _OFM_NUM), \
 osv, isv)
inline uint get_g_os_zy_is_x_osv_isv_index(uint g, uint o, uint i, uint z, uint y, uint x,
 uint o_size, uint i_size, uint z_size, uint y_size, uint x_size,
 uint osv, uint isv) {
 uint is_size = (i_size + isv - 1) / isv;
 uint os_size = (o_size + osv - 1) / osv;
 uint isv_index = i % isv;
 uint osv_index = o % osv;
 uint is_index = i / isv;
 uint os_index = o / osv;
 uint isv_pitch = 1;
 uint osv_pitch = isv_pitch * isv;
 uint x_pitch = osv_pitch * osv;
 uint is_pitch = x_pitch * x_size;
 uint y_pitch = is_pitch * is_size;
 uint z_pitch = y_pitch * y_size;
 uint os_pitch = z_pitch * z_size;
 uint g_pitch = os_pitch * os_size;
 uint index = 0;
 index += isv_index * isv_pitch;
 index += osv_index * osv_pitch;
 index += is_index * is_pitch;
 index += x * x_pitch;
 index += y * y_pitch;
 index += z * z_pitch;
 index += os_index * os_pitch;
 index += g * g_pitch;
 return index;
}
#define GET_FILTER_G_OS_ZY_IS_X_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, osv, isv) \
 get_g_os_zy_is_x_osv_isv_index( \
 g, o, i, z, y, x, \
 CAT(tensor, _OFM_NUM), \
 CAT(tensor, _IFM_NUM), \
 CAT(tensor, _SIZE_Z), \
 CAT(tensor, _SIZE_Y), \
 CAT(tensor, _SIZE_X), \
 osv, isv)

inline int imad_SW(int acc, uchar4 input, char4 weight) __attribute__((overloadable)) {
 acc += input[0] * weight[0];
 acc += input[1] * weight[1];
 acc += input[2] * weight[2];
 acc += input[3] * weight[3];
 return acc;
}
inline int imad_SW(int acc, char4 input, char4 weight) __attribute__((overloadable)) {
 acc += input[0] * weight[0];
 acc += input[1] * weight[1];
 acc += input[2] * weight[2];
 acc += input[3] * weight[3];
 return acc;
}
inline int imad_SW(int acc, char4 input, uchar4 weight) __attribute__((overloadable)) {
 acc += input[0] * weight[0];
 acc += input[1] * weight[1];
 acc += input[2] * weight[2];
 acc += input[3] * weight[3];
 return acc;
}
inline int imad_SW(int acc, uchar4 input, uchar4 weight) __attribute__((overloadable)) {
 acc += input[0] * weight[0];
 acc += input[1] * weight[1];
 acc += input[2] * weight[2];
 acc += input[3] * weight[3];
 return acc;
}
#define IMAD(_O, _I, _W) imad_SW(_O, _I, _W)

inline float convert_as_int4_float(uchar source, uint index) {
 float out;
 if (index % 2 == 0) {
 out = source & 0xF;
 } else {
 out = source >> 4;
 }
 if (out > 7.f)
 out -= 16.f;
 return out;
}
inline float convert_as_uint4_float(uchar source, uint index) {
 float out;
 if (index % 2 == 0) {
 out = source & 0xF;
 } else {
 out = source >> 4;
 }
 return out;
}

typedef struct __attribute__ ((packed)) int4x2_t { char s0; } int4x2_t;
typedef struct __attribute__ ((packed)) int4x4_t { int4x2_t s0; int4x2_t s1; } int4x4_t;
typedef struct __attribute__ ((packed)) int4x8_t { int4x2_t s0; int4x2_t s1; int4x2_t s2; int4x2_t s3; } int4x8_t;
typedef struct __attribute__ ((packed)) int4x16_t { int4x2_t s0; int4x2_t s1; int4x2_t s2; int4x2_t s3; int4x2_t s4; int4x2_t s5; int4x2_t s6; int4x2_t s7; } int4x16_t;
typedef struct __attribute__ ((packed)) uint4x2_t { uchar s0; } uint4x2_t;
typedef struct __attribute__ ((packed)) uint4x4_t { uint4x2_t s0; uint4x2_t s1; } uint4x4_t;
typedef struct __attribute__ ((packed)) uint4x8_t { uint4x2_t s0; uint4x2_t s1; uint4x2_t s2; uint4x2_t s3; } uint4x8_t;
typedef struct __attribute__ ((packed)) uint4x16_t { uint4x2_t s0; uint4x2_t s1; uint4x2_t s2; uint4x2_t s3; uint4x2_t s4; uint4x2_t s5; uint4x2_t s6; uint4x2_t s7; } uint4x16_t;
inline uchar2 cvt_uint4x2_to_uint8x2(uint4x2_t v) __attribute__((overloadable)) {
 const uchar v0 = v.s0 & 0x0F;
 const uchar v1 = (v.s0 & 0xF0) >> 4;
 return (uchar2)(v0, v1);
}
inline char2 cvt_uint4x2_to_int8x2(uint4x2_t v) __attribute__((overloadable)) {
 const char v0 = convert_char(v.s0 & 0x0F);
 const char v1 = convert_char((v.s0 & 0xF0) >> 4);
 return (char2)(v0, v1);
}
inline char2 cvt_int4x2_to_int8x2(int4x2_t v) __attribute__((overloadable)) {
 const char s_bit = (v.s0 & convert_char(0x08));
 const char mask = s_bit > 0 ? convert_char(0xF0) : convert_char(0x00);
 const char v0 = (v.s0 & convert_char(0x0F)) | mask;
 const char v1 = v.s0 >> 4;
 return (char2)(v0, v1);
}
inline uchar2 unpack_to_uchar(uint4x2_t v) __attribute__((overloadable)) {
 return cvt_uint4x2_to_uint8x2(v);
}
inline char2 unpack_to_char(int4x2_t v) __attribute__((overloadable)) {
 return cvt_int4x2_to_int8x2(v);
}
inline char2 unpack_to_char(uint4x2_t v) __attribute__((overloadable)) {
 return convert_char2(cvt_uint4x2_to_uint8x2(v));
}
inline char4 unpack_to_char(int4x4_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s1);
 return (char4)(v0.s0, v0.s1, v1.s0, v1.s1);
}
inline char4 unpack_to_char(uint4x4_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s1);
 return (char4)(v0.s0, v0.s1, v1.s0, v1.s1);
}
inline uchar4 unpack_to_uchar(uint4x4_t v) __attribute__((overloadable)) {
 uchar2 v0 = unpack_to_uchar(v.s0);
 uchar2 v1 = unpack_to_uchar(v.s1);
 return (uchar4)(v0.s0, v0.s1, v1.s0, v1.s1);
}
inline char4 unpack_transposed_to_char(int4x4_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s1);
 return (char4)(v0.s0, v1.s0, v0.s1, v1.s1);
}
inline char4 unpack_transposed_to_char(uint4x4_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s1);
 return (char4)(v0.s0, v1.s0, v0.s1, v1.s1);
}
inline uchar4 unpack_transposed_to_uchar(uint4x4_t v) __attribute__((overloadable)) {
 uchar2 v0 = unpack_to_uchar(v.s0);
 uchar2 v1 = unpack_to_uchar(v.s1);
 return (uchar4)(v0.s0, v1.s0, v0.s1, v1.s1);
}
inline uchar8 unpack_to_uchar(uint4x8_t v) __attribute__((overloadable)) {
 uchar2 v0 = unpack_to_uchar(v.s0);
 uchar2 v1 = unpack_to_uchar(v.s1);
 uchar2 v2 = unpack_to_uchar(v.s2);
 uchar2 v3 = unpack_to_uchar(v.s3);
 return (uchar8)(v0.s0, v0.s1, v1.s0, v1.s1, v2.s0, v2.s1, v3.s0, v3.s1);
}
inline char8 unpack_to_char(int4x8_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s1);
 char2 v2 = unpack_to_char(v.s2);
 char2 v3 = unpack_to_char(v.s3);
 return (char8)(v0.s0, v0.s1, v1.s0, v1.s1, v2.s0, v2.s1, v3.s0, v3.s1);
}
inline char8 unpack_to_char(uint4x8_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s1);
 char2 v2 = unpack_to_char(v.s2);
 char2 v3 = unpack_to_char(v.s3);
 return (char8)(v0.s0, v0.s1, v1.s0, v1.s1, v2.s0, v2.s1, v3.s0, v3.s1);
}
inline char8 unpack_transposed_to_char(int4x8_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s1);
 char2 v2 = unpack_to_char(v.s2);
 char2 v3 = unpack_to_char(v.s3);
 return (char8)(v0.s0, v1.s0, v2.s0, v3.s0, v0.s1, v1.s1, v2.s1, v3.s1);
}
inline char8 unpack_transposed_to_char(uint4x8_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s1);
 char2 v2 = unpack_to_char(v.s2);
 char2 v3 = unpack_to_char(v.s3);
 return (char8)(v0.s0, v1.s0, v2.s0, v3.s0, v0.s1, v1.s1, v2.s1, v3.s1);
}
inline uchar8 unpack_transposed_to_uchar(uint4x8_t v) __attribute__((overloadable)) {
 uchar2 v0 = unpack_to_uchar(v.s0);
 uchar2 v1 = unpack_to_uchar(v.s1);
 uchar2 v2 = unpack_to_uchar(v.s2);
 uchar2 v3 = unpack_to_uchar(v.s3);
 return (uchar8)(v0.s0, v1.s0, v2.s0, v3.s0, v0.s1, v1.s1, v2.s1, v3.s1);
}
inline float2 unpack_to_float(uint4x2_t v) __attribute__((overloadable)) {
 return convert_float2(cvt_uint4x2_to_uint8x2(v));
}
inline float2 unpack_to_float(int4x2_t v) __attribute__((overloadable)) {
 return convert_float2(cvt_int4x2_to_int8x2(v));
}
inline float4 unpack_to_float(uint4x4_t v) __attribute__((overloadable)) {
 float2 f0 = unpack_to_float(v.s0);
 float2 f1 = unpack_to_float(v.s1);
 return (float4)(f0.s0, f0.s1, f1.s0, f1.s1);
}
inline float4 unpack_to_float(int4x4_t v) __attribute__((overloadable)) {
 float2 f0 = unpack_to_float(v.s0);
 float2 f1 = unpack_to_float(v.s1);
 return (float4)(f0.s0, f0.s1, f1.s0, f1.s1);
}
inline float8 unpack_to_float(uint4x8_t v) __attribute__((overloadable)) {
 float2 f0 = unpack_to_float(v.s0);
 float2 f1 = unpack_to_float(v.s1);
 float2 f2 = unpack_to_float(v.s2);
 float2 f3 = unpack_to_float(v.s3);
 return (float8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}
inline float8 unpack_to_float(int4x8_t v) __attribute__((overloadable)) {
 float2 f0 = unpack_to_float(v.s0);
 float2 f1 = unpack_to_float(v.s1);
 float2 f2 = unpack_to_float(v.s2);
 float2 f3 = unpack_to_float(v.s3);
 return (float8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}
#if defined(cl_khr_fp16)
inline half2 unpack_to_half(uint4x2_t v) __attribute__((overloadable)) {
 return convert_half2(cvt_uint4x2_to_uint8x2(v));
}
inline half2 unpack_to_half(int4x2_t v) __attribute__((overloadable)) {
 return convert_half2(cvt_int4x2_to_int8x2(v));
}
inline half4 unpack_to_half(uint4x4_t v) __attribute__((overloadable)) {
 half2 f0 = unpack_to_half(v.s0);
 half2 f1 = unpack_to_half(v.s1);
 return (half4)(f0.s0, f0.s1, f1.s0, f1.s1);
}
inline half4 unpack_to_half_osv32_isv2(uint4x4_t v) __attribute__((overloadable)) {
 half2 f0 = unpack_to_half(v.s0);
 half2 f1 = unpack_to_half(v.s1);
 return (half4)(f0.s0, f0.s1, f1.s0, f1.s1);
}
inline half4 unpack_to_half(int4x4_t v) __attribute__((overloadable)) {
 half2 f0 = unpack_to_half(v.s0);
 half2 f1 = unpack_to_half(v.s1);
 return (half4)(f0.s0, f0.s1, f1.s0, f1.s1);
}

inline half4 unpack_to_half_osv32_isv2(int4x4_t v) __attribute__((overloadable)) {
 half2 f0 = unpack_to_half(v.s0);
 half2 f1 = unpack_to_half(v.s1);
 return (half4)(f0.s0, f0.s1, f1.s0, f1.s1);
}
inline half8 unpack_to_half(uint4x8_t v) __attribute__((overloadable)) {
 half2 f0 = unpack_to_half(v.s0);
 half2 f1 = unpack_to_half(v.s1);
 half2 f2 = unpack_to_half(v.s2);
 half2 f3 = unpack_to_half(v.s3);
 return (half8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}
inline half8 unpack_to_half_osv32_isv2(uint4x8_t v) __attribute__((overloadable)) {
 half2 f0 = unpack_to_half(v.s0);
 half2 f1 = unpack_to_half(v.s2);
 half2 f2 = unpack_to_half(v.s1);
 half2 f3 = unpack_to_half(v.s3);
 return (half8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}
inline half8 unpack_to_half(int4x8_t v) __attribute__((overloadable)) {
 half2 f0 = unpack_to_half(v.s0);
 half2 f1 = unpack_to_half(v.s1);
 half2 f2 = unpack_to_half(v.s2);
 half2 f3 = unpack_to_half(v.s3);
 return (half8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}
inline half8 unpack_to_half_osv32_isv2(int4x8_t v) __attribute__((overloadable)) {
 half2 f0 = unpack_to_half(v.s0);
 half2 f1 = unpack_to_half(v.s2);
 half2 f2 = unpack_to_half(v.s1);
 half2 f3 = unpack_to_half(v.s3);
 return (half8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}
inline char8 unpack_to_char_osv32_isv2(int4x8_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s2);
 char2 v2 = unpack_to_char(v.s1);
 char2 v3 = unpack_to_char(v.s3);
 return (char8)(v0.s0, v0.s1, v1.s0, v1.s1, v2.s0, v2.s1, v3.s0, v3.s1);
}
inline char8 unpack_to_char_osv32_isv2(uint4x8_t v) __attribute__((overloadable)) {
 char2 v0 = unpack_to_char(v.s0);
 char2 v1 = unpack_to_char(v.s2);
 char2 v2 = unpack_to_char(v.s1);
 char2 v3 = unpack_to_char(v.s3);
 return (char8)(v0.s0, v0.s1, v1.s0, v1.s1, v2.s0, v2.s1, v3.s0, v3.s1);
}
inline uchar8 unpack_to_uchar_osv32_isv2(uint4x8_t v) __attribute__((overloadable)) {
 uchar2 v0 = unpack_to_uchar(v.s0);
 uchar2 v1 = unpack_to_uchar(v.s2);
 uchar2 v2 = unpack_to_uchar(v.s1);
 uchar2 v3 = unpack_to_uchar(v.s3);
 return (uchar8)(v0.s0, v0.s1, v1.s0, v1.s1, v2.s0, v2.s1, v3.s0, v3.s1);
}
#endif
#define UNPACK_INT4x2(target_type, value) CAT(unpack_to_, target_type)(value)
#define UNPACK_INT4x2_OSV32_ISV2(target_type, value) CAT(CAT(unpack_to_, target_type), _osv32_isv2)(value)
#define UNPACK_INT4x4_OSV32_ISV2(target_type, value) CAT(CAT(unpack_to_, target_type), _osv32_isv2)(value)
#define UNPACK_TRANSPOSED_INT4x2(target_type, value) CAT(unpack_transposed_to_, target_type)(value)

#define BLOCK_READ_TYPE_size1 uchar
#define BLOCK_READ_TYPE_size2 ushort
#define BLOCK_READ_TYPE_size4 uint
#define BLOCK_READ_TYPE_size8 ulong
#define BLOCK_READ_TYPE(type_size) CAT(BLOCK_READ_TYPE_size, type_size)
#define BLOCK_READ_FUNC_size1 _sub_group_block_read_uc
#define BLOCK_READ_FUNC_size2 _sub_group_block_read_us
#define BLOCK_READ_FUNC_size4 _sub_group_block_read
#define BLOCK_READ_FUNC_size8 _sub_group_block_read_ul
#define BLOCK_READ_FUNC(type_size) CAT(BLOCK_READ_FUNC_size, type_size)
#define BLOCK_READN_FUNC_SIZE_DEF(type_size, vector_size) MAKE_VECTOR_TYPE(BLOCK_READ_FUNC(type_size), vector_size)
#define BLOCK_READN_FUNC_size1(vector_size) BLOCK_READN_FUNC_SIZE_DEF(1, vector_size)
#define BLOCK_READN_FUNC_size2(vector_size) BLOCK_READN_FUNC_SIZE_DEF(2, vector_size)
#define BLOCK_READN_FUNC_size4(vector_size) BLOCK_READN_FUNC_SIZE_DEF(4, vector_size)
#define BLOCK_READN_FUNC_size8(vector_size) BLOCK_READN_FUNC_SIZE_DEF(8, vector_size)
#define BLOCK_READN_FUNC(type_size, vector_size) CAT(BLOCK_READN_FUNC_size, type_size)(vector_size)
#define BLOCK_READN_RAW(type_size, vector_size, addr_space, ptr, offset) \
 BLOCK_READN_FUNC(type_size, vector_size)((const addr_space BLOCK_READ_TYPE(type_size)*)(ptr) + (offset))
#define BLOCK_READN(type, vector_size, ptr, offset) \
 AS_TYPE(MAKE_VECTOR_TYPE(type, vector_size), BLOCK_READN_RAW(TYPE_SIZE(type), vector_size, __global, ptr, offset))
#define BLOCK_READN_SLM(type, vector_size, ptr, offset) \
 AS_TYPE(MAKE_VECTOR_TYPE(type, vector_size), BLOCK_READN_RAW(TYPE_SIZE(type), vector_size, __local, ptr, offset))
#define DT_INPUT_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)
#define DT_INPUT_BLOCK_READ2(ptr, offset) BLOCK_READN(INPUT0_TYPE, 2, ptr, offset)
#define DT_INPUT_BLOCK_READ4(ptr, offset) BLOCK_READN(INPUT0_TYPE, 4, ptr, offset)
#define DT_INPUT_BLOCK_READ8(ptr, offset) BLOCK_READN(INPUT0_TYPE, 8, ptr, offset)
#define DT_INPUT_BLOCK_READ16(ptr, offset) BLOCK_READN(INPUT0_TYPE, 16, ptr, offset)
#define DT_BIAS_BLOCK_READ(ptr, offset) BLOCK_READN(BIAS_TYPE, 1, ptr, offset)
#define DT_BIAS_BLOCK_READ2(ptr, offset) BLOCK_READN(BIAS_TYPE, 2, ptr, offset)
#define DT_BIAS_BLOCK_READ4(ptr, offset) BLOCK_READN(BIAS_TYPE, 4, ptr, offset)
#define DT_BIAS_BLOCK_READ8(ptr, offset) BLOCK_READN(BIAS_TYPE, 8, ptr, offset)
#define DT_BIAS_BLOCK_READ16(ptr, offset) BLOCK_READN(BIAS_TYPE, 16, ptr, offset)
#define DT_FILTER_BLOCK_READ(ptr, offset) BLOCK_READN(FILTER_TYPE, 1, ptr, offset)
#define DT_FILTER_BLOCK_READ2(ptr, offset) BLOCK_READN(FILTER_TYPE, 2, ptr, offset)
#define DT_FILTER_BLOCK_READ4(ptr, offset) BLOCK_READN(FILTER_TYPE, 4, ptr, offset)
#define DT_FILTER_BLOCK_READ8(ptr, offset) BLOCK_READN(FILTER_TYPE, 8, ptr, offset)
#define DT_FILTER_BLOCK_READ16(ptr, offset) BLOCK_READN(FILTER_TYPE, 16, ptr, offset)
#define BLOCK_READ_IMPL_1 ret = ptr[idx];
#define BLOCK_READ_IMPL_2 \
 ret.s0 = ptr[idx]; idx += get_max_sub_group_size(); \
 ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
#define BLOCK_READ_IMPL_4 \
 BLOCK_READ_IMPL_2 \
 ret.s2 = ptr[idx]; idx += get_max_sub_group_size(); \
 ret.s3 = ptr[idx]; idx += get_max_sub_group_size();
#define BLOCK_READ_IMPL_8 \
 BLOCK_READ_IMPL_4 \
 ret.s4 = ptr[idx]; idx += get_max_sub_group_size(); \
 ret.s5 = ptr[idx]; idx += get_max_sub_group_size(); \
 ret.s6 = ptr[idx]; idx += get_max_sub_group_size(); \
 ret.s7 = ptr[idx]; idx += get_max_sub_group_size();
#define BLOCK_READ_IMPL_16 \
 BLOCK_READ_IMPL_8 \
 ret.s8 = ptr[idx]; idx += get_max_sub_group_size(); \
 ret.s9 = ptr[idx]; idx += get_max_sub_group_size(); \
 ret.sa = ptr[idx]; idx += get_max_sub_group_size(); \
 ret.sb = ptr[idx]; idx += get_max_sub_group_size(); \
 ret.sc = ptr[idx]; idx += get_max_sub_group_size(); \
 ret.sd = ptr[idx]; idx += get_max_sub_group_size(); \
 ret.se = ptr[idx]; idx += get_max_sub_group_size(); \
 ret.sf = ptr[idx]; idx += get_max_sub_group_size();
#define BLOCK_READ_IMPL(vec_size) CAT(BLOCK_READ_IMPL_, vec_size)
#define BLOCK_READ_FUNC_NAME(type_size, vec_size) MAKE_VECTOR_TYPE(BLOCK_READ_FUNC(type_size), vec_size)
#define DECLARE_BLOCK_READ_EMULATION(type_size, vec_size) \
 inline MAKE_VECTOR_TYPE(BLOCK_READ_TYPE(type_size), vec_size) BLOCK_READ_FUNC_NAME(type_size, vec_size)(const __global BLOCK_READ_TYPE(type_size)* ptr) { \
 uint idx = get_sub_group_local_id(); \
 MAKE_VECTOR_TYPE(BLOCK_READ_TYPE(type_size), vec_size) ret; \
 BLOCK_READ_IMPL(vec_size) \
 return ret; \
}
#if defined(cl_intel_subgroups)
 #define _sub_group_block_read(ptr) intel_sub_group_block_read(ptr)
 #define _sub_group_block_read2(ptr) intel_sub_group_block_read2(ptr)
 #define _sub_group_block_read4(ptr) intel_sub_group_block_read4(ptr)
 #define _sub_group_block_read8(ptr) intel_sub_group_block_read8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
 DECLARE_BLOCK_READ_EMULATION(4, 1)
 DECLARE_BLOCK_READ_EMULATION(4, 2)
 DECLARE_BLOCK_READ_EMULATION(4, 4)
 DECLARE_BLOCK_READ_EMULATION(4, 8)
#endif
#if defined(cl_intel_subgroups_short)
 #define _sub_group_block_read_us(ptr) intel_sub_group_block_read_us(ptr)
 #define _sub_group_block_read_us2(ptr) intel_sub_group_block_read_us2(ptr)
 #define _sub_group_block_read_us4(ptr) intel_sub_group_block_read_us4(ptr)
 #define _sub_group_block_read_us8(ptr) intel_sub_group_block_read_us8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
 DECLARE_BLOCK_READ_EMULATION(2, 1)
 DECLARE_BLOCK_READ_EMULATION(2, 2)
 DECLARE_BLOCK_READ_EMULATION(2, 4)
 DECLARE_BLOCK_READ_EMULATION(2, 8)
#endif
#if defined(cl_intel_subgroups_char)
 #define _sub_group_block_read_uc(ptr) intel_sub_group_block_read_uc(ptr)
 #define _sub_group_block_read_uc2(ptr) intel_sub_group_block_read_uc2(ptr)
 #define _sub_group_block_read_uc4(ptr) intel_sub_group_block_read_uc4(ptr)
 #define _sub_group_block_read_uc8(ptr) intel_sub_group_block_read_uc8(ptr)
 #define _sub_group_block_read_uc16(ptr) intel_sub_group_block_read_uc16(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
 DECLARE_BLOCK_READ_EMULATION(1, 1)
 DECLARE_BLOCK_READ_EMULATION(1, 2)
 DECLARE_BLOCK_READ_EMULATION(1, 4)
 DECLARE_BLOCK_READ_EMULATION(1, 8)
 DECLARE_BLOCK_READ_EMULATION(1, 16)
#endif
#if defined(cl_intel_subgroups_long)
 #define _sub_group_block_read_ul(ptr) intel_sub_group_block_read_ul(ptr)
 #define _sub_group_block_read_ul2(ptr) intel_sub_group_block_read_ul2(ptr)
 #define _sub_group_block_read_ul4(ptr) intel_sub_group_block_read_ul4(ptr)
 #define _sub_group_block_read_ul8(ptr) intel_sub_group_block_read_ul8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
 DECLARE_BLOCK_READ_EMULATION(8, 1)
 DECLARE_BLOCK_READ_EMULATION(8, 2)
 DECLARE_BLOCK_READ_EMULATION(8, 4)
 DECLARE_BLOCK_READ_EMULATION(8, 8)
#endif

#define BLOCK_WRITE_TYPE_size1 uchar
#define BLOCK_WRITE_TYPE_size2 ushort
#define BLOCK_WRITE_TYPE_size4 uint
#define BLOCK_WRITE_TYPE_size8 ulong
#define BLOCK_WRITE_TYPE(type_size) CAT(BLOCK_WRITE_TYPE_size, type_size)
#define BLOCK_WRITE_FUNC_size1 _sub_group_block_write_uc
#define BLOCK_WRITE_FUNC_size2 _sub_group_block_write_us
#define BLOCK_WRITE_FUNC_size4 _sub_group_block_write
#define BLOCK_WRITE_FUNC_size8 _sub_group_block_write_ul
#define BLOCK_WRITE_FUNC(type_size) CAT(BLOCK_WRITE_FUNC_size, type_size)
#define BLOCK_WRITEN_FUNC_SIZE_DEF(type_size, vector_size) MAKE_VECTOR_TYPE(BLOCK_WRITE_FUNC(type_size), vector_size)
#define BLOCK_WRITEN_FUNC_size1(vector_size) BLOCK_WRITEN_FUNC_SIZE_DEF(1, vector_size)
#define BLOCK_WRITEN_FUNC_size2(vector_size) BLOCK_WRITEN_FUNC_SIZE_DEF(2, vector_size)
#define BLOCK_WRITEN_FUNC_size4(vector_size) BLOCK_WRITEN_FUNC_SIZE_DEF(4, vector_size)
#define BLOCK_WRITEN_FUNC_size8(vector_size) BLOCK_WRITEN_FUNC_SIZE_DEF(8, vector_size)
#define BLOCK_WRITEN_FUNC(type_size, vector_size) CAT(BLOCK_WRITEN_FUNC_size, type_size)(vector_size)
#define BLOCK_WRITEN_RAW(type_size, vector_size, addr_space, ptr, offset, val) \
 BLOCK_WRITEN_FUNC(type_size, vector_size)( \
 (addr_space BLOCK_WRITE_TYPE(type_size)*)(ptr) + (offset), \
 AS_TYPE(MAKE_VECTOR_TYPE(BLOCK_WRITE_TYPE(type_size), vector_size), val))
#define BLOCK_WRITEN(type, vector_size, ptr, offset, val) \
 BLOCK_WRITEN_RAW(TYPE_SIZE(type), vector_size, __global, ptr, offset, val)
#define BLOCK_WRITEN_SLM(type, vector_size, ptr, offset, val) \
 BLOCK_WRITEN_RAW(TYPE_SIZE(type), vector_size, __local, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, 1, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE2(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, 2, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE4(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, 4, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE8(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, 8, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE16(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, 16, ptr, offset, val)
#define BLOCK_WRITE_IMPL_1 out_ptr[idx] = v;
#define BLOCK_WRITE_IMPL_2 \
 out_ptr[idx] = v.s0; idx += get_max_sub_group_size(); \
 out_ptr[idx] = v.s1; idx += get_max_sub_group_size();
#define BLOCK_WRITE_IMPL_4 \
 BLOCK_WRITE_IMPL_2 \
 out_ptr[idx] = v.s2; idx += get_max_sub_group_size(); \
 out_ptr[idx] = v.s3; idx += get_max_sub_group_size();
#define BLOCK_WRITE_IMPL_8 \
 BLOCK_WRITE_IMPL_4 \
 out_ptr[idx] = v.s4; idx += get_max_sub_group_size(); \
 out_ptr[idx] = v.s5; idx += get_max_sub_group_size(); \
 out_ptr[idx] = v.s6; idx += get_max_sub_group_size(); \
 out_ptr[idx] = v.s7; idx += get_max_sub_group_size();
#define BLOCK_WRITE_IMPL_16 \
 BLOCK_WRITE_IMPL_8 \
 out_ptr[idx] = v.s8; idx += get_max_sub_group_size(); \
 out_ptr[idx] = v.s9; idx += get_max_sub_group_size(); \
 out_ptr[idx] = v.sa; idx += get_max_sub_group_size(); \
 out_ptr[idx] = v.sb; idx += get_max_sub_group_size(); \
 out_ptr[idx] = v.sc; idx += get_max_sub_group_size(); \
 out_ptr[idx] = v.sd; idx += get_max_sub_group_size(); \
 out_ptr[idx] = v.se; idx += get_max_sub_group_size(); \
 out_ptr[idx] = v.sf; idx += get_max_sub_group_size();
#define BLOCK_WRITE_IMPL(vec_size) CAT(BLOCK_WRITE_IMPL_, vec_size)
#define BLOCK_WRITE_FUNC_NAME(type_size, vec_size) MAKE_VECTOR_TYPE(BLOCK_WRITE_FUNC(type_size), vec_size)
#define DECLARE_BLOCK_WRITE_EMULATION(type_size, vec_size) \
 inline void BLOCK_WRITE_FUNC_NAME(type_size, vec_size)(__global BLOCK_WRITE_TYPE(type_size)* out_ptr, \
 MAKE_VECTOR_TYPE(BLOCK_WRITE_TYPE(type_size), vec_size) v) { \
 uint idx = get_sub_group_local_id(); \
 BLOCK_WRITE_IMPL(vec_size) \
}
#if defined(cl_intel_subgroups)
 #define _sub_group_block_write(ptr, v) intel_sub_group_block_write(ptr, v)
 #define _sub_group_block_write2(ptr, v) intel_sub_group_block_write2(ptr, v)
 #define _sub_group_block_write4(ptr, v) intel_sub_group_block_write4(ptr, v)
 #define _sub_group_block_write8(ptr, v) intel_sub_group_block_write8(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
 DECLARE_BLOCK_WRITE_EMULATION(4, 1)
 DECLARE_BLOCK_WRITE_EMULATION(4, 2)
 DECLARE_BLOCK_WRITE_EMULATION(4, 4)
 DECLARE_BLOCK_WRITE_EMULATION(4, 8)
#endif
#if defined(cl_intel_subgroups_short)
 #define _sub_group_block_write_us(ptr, v) intel_sub_group_block_write_us(ptr, v)
 #define _sub_group_block_write_us2(ptr, v) intel_sub_group_block_write_us2(ptr, v)
 #define _sub_group_block_write_us4(ptr, v) intel_sub_group_block_write_us4(ptr, v)
 #define _sub_group_block_write_us8(ptr, v) intel_sub_group_block_write_us8(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
 DECLARE_BLOCK_WRITE_EMULATION(2, 1)
 DECLARE_BLOCK_WRITE_EMULATION(2, 2)
 DECLARE_BLOCK_WRITE_EMULATION(2, 4)
 DECLARE_BLOCK_WRITE_EMULATION(2, 8)
#endif
#if defined(cl_intel_subgroups_char)
 #define _sub_group_block_write_uc(ptr, v) intel_sub_group_block_write_uc(ptr, v)
 #define _sub_group_block_write_uc2(ptr, v) intel_sub_group_block_write_uc2(ptr, v)
 #define _sub_group_block_write_uc4(ptr, v) intel_sub_group_block_write_uc4(ptr, v)
 #define _sub_group_block_write_uc8(ptr, v) intel_sub_group_block_write_uc8(ptr, v)
 #define _sub_group_block_write_uc16(ptr, v) intel_sub_group_block_write_uc16(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
 DECLARE_BLOCK_WRITE_EMULATION(1, 1)
 DECLARE_BLOCK_WRITE_EMULATION(1, 2)
 DECLARE_BLOCK_WRITE_EMULATION(1, 4)
 DECLARE_BLOCK_WRITE_EMULATION(1, 8)
 DECLARE_BLOCK_WRITE_EMULATION(1, 16)
#endif
#if defined(cl_intel_subgroups_long)
 #define _sub_group_block_write_ul(ptr, v) intel_sub_group_block_write_ul(ptr, v)
 #define _sub_group_block_write_ul2(ptr, v) intel_sub_group_block_write_ul2(ptr, v)
 #define _sub_group_block_write_ul4(ptr, v) intel_sub_group_block_write_ul4(ptr, v)
 #define _sub_group_block_write_ul8(ptr, v) intel_sub_group_block_write_ul8(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
 DECLARE_BLOCK_WRITE_EMULATION(8, 1)
 DECLARE_BLOCK_WRITE_EMULATION(8, 2)
 DECLARE_BLOCK_WRITE_EMULATION(8, 4)
 DECLARE_BLOCK_WRITE_EMULATION(8, 8)
#endif

#ifdef cl_intel_subgroups
#define _sub_group_shuffle(v, c) intel_sub_group_shuffle(v, c)
#define _sub_group_shuffle_up(c, n, d) intel_sub_group_shuffle_up(c, n, d)
#define _sub_group_shuffle_down(c, n, d) intel_sub_group_shuffle_down(c, n, d)
#elif (__OPENCL_C_VERSION__ >= 200)
#define DECLARE_SUB_GROUP_SHUFFLE1(type, cast_type) \
inline type _sub_group_shuffle(type v, uint c) __attribute__((overloadable)) { \
 return AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v), c)); \
}
#define DECLARE_SUB_GROUP_SHUFFLE2(type, cast_type) \
inline CAT(type, 2) _sub_group_shuffle(CAT(type, 2) v, uint c) __attribute__((overloadable)) { \
 return (CAT(type, 2))( AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s0), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s1), c))); \
}
#define DECLARE_SUB_GROUP_SHUFFLE4(type, cast_type) \
inline CAT(type, 4) _sub_group_shuffle(CAT(type, 4) v, uint c) __attribute__((overloadable)) { \
 return (CAT(type, 4))( AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s0), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s1), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s2), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s3), c))); \
}
#define DECLARE_SUB_GROUP_SHUFFLE8(type, cast_type) \
inline CAT(type, 8) _sub_group_shuffle(CAT(type, 8) v, uint c) __attribute__((overloadable)) { \
 return (CAT(type, 8))( AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s0), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s1), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s2), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s3), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s4), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s5), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s6), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s7), c))); \
}
#define DECLARE_SUB_GROUP_SHUFFLE16(type, cast_type) \
inline CAT(type, 16) _sub_group_shuffle(CAT(type, 16) v, uint c) __attribute__((overloadable)) { \
 return (CAT(type, 16))( AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s0), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s1), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s2), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s3), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s4), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s5), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s6), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s7), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s8), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s9), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sa), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sb), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sc), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sd), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.se), c)), \
 AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sf), c))); \
}
#define DECLARE_SUB_GROUP_SHUFFLE(type) \
 DECLARE_SUB_GROUP_SHUFFLE1(type, type) \
 DECLARE_SUB_GROUP_SHUFFLE2(type, type) \
 DECLARE_SUB_GROUP_SHUFFLE4(type, type) \
 DECLARE_SUB_GROUP_SHUFFLE8(type, type) \
 DECLARE_SUB_GROUP_SHUFFLE16(type, type)
#define DECLARE_SUB_GROUP_SHUFFLE_CASTED(type, cast_type) \
 DECLARE_SUB_GROUP_SHUFFLE1(type, cast_type) \
 DECLARE_SUB_GROUP_SHUFFLE2(type, cast_type) \
 DECLARE_SUB_GROUP_SHUFFLE4(type, cast_type) \
 DECLARE_SUB_GROUP_SHUFFLE8(type, cast_type) \
 DECLARE_SUB_GROUP_SHUFFLE16(type, cast_type)
DECLARE_SUB_GROUP_SHUFFLE(int)
DECLARE_SUB_GROUP_SHUFFLE(uint)
DECLARE_SUB_GROUP_SHUFFLE(float)
#if defined(cl_khr_fp16)
 DECLARE_SUB_GROUP_SHUFFLE(half)
 DECLARE_SUB_GROUP_SHUFFLE_CASTED(short, half)
 DECLARE_SUB_GROUP_SHUFFLE_CASTED(ushort, half)
#endif
#endif

typedef struct half5 { half s0; half s1; half s2; half s3; half s4; } half5;
typedef struct half6 { half s0; half s1; half s2; half s3; half s4; half s5; } half6;
typedef struct half7 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; } half7;
typedef struct half9 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7;
 half s8; } half9;
typedef struct half10 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7;
 half s8; half s9; } half10;
typedef struct half11 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7;
 half s8; half s9; half sa; } half11;
typedef struct half12 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7;
 half s8; half s9; half sa; half sb;} half12;
typedef struct half13 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7;
 half s8; half s9; half sa; half sb; half sc;} half13;
typedef struct half14 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7;
 half s8; half s9; half sa; half sb; half sc; half se;} half14;
typedef struct half15 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7;
 half s8; half s9; half sa; half sb; half sc; half se; half sf;} half15;
typedef struct half0 { half s0; } half0;
typedef struct float5 { float s0; float s1; float s2; float s3; float s4; } float5;
typedef struct float6 { float s0; float s1; float s2; float s3; float s4; float s5; } float6;
typedef struct float7 { float s0; float s1; float s2; float s3; float s4; float s5; float s6; } float7;
typedef struct float9 { float s0; float s1; float s2; float s3; float s4; float s5; float s6; float s7; float s8; } float9;
typedef struct float10 { float s0; float s1; float s2; float s3; float s4; float s5;
 float s6; float s7; float s8; float s9;} float10;
typedef struct float11 { float s0; float s1; float s2; float s3; float s4; float s5;
 float s6; float s7; float s8; float s9; float sa;} float11;
typedef struct float12 { float s0; float s1; float s2; float s3; float s4; float s5;
 float s6; float s7; float s8; float s9; float sa; float sb; } float12;
typedef struct float13 { float s0; float s1; float s2; float s3; float s4; float s5;
 float s6; float s7; float s8; float s9; float sa; float sb; float sc;} float13;
typedef struct float14 { float s0; float s1; float s2; float s3; float s4; float s5;
 float s6; float s7; float s8; float s9; float sa; float sb; float sc; float sd; } float14;
typedef struct float15 { float s0; float s1; float s2; float s3; float s4; float s5;
 float s6; float s7; float s8; float s9; float sa; float sb; float sc; float sd; float se; } float15;
typedef struct float0 { float s0; } float0;

//====================================================
// Kernel template: pooling_gpu_blocked 
// Kernel name: pooling_gpu_blocked_17718859278101007469_0_0
#define KERNEL(name) __kernel void pooling_gpu_blocked_17718859278101007469_0_0
#define KERNEL_ID pooling_gpu_blocked_17718859278101007469_0_0
#define FUNC(name)  _##name##_pooling_gpu_blocked_17718859278101007469_0_0
#define FUNC_CALL(name)  _##name##_pooling_gpu_blocked_17718859278101007469_0_0
#define CONST_ARRAY_DECL(name) __constant size_t  _##name##_pooling_gpu_blocked_17718859278101007469_0_0 []
#define CONST_ARRAY_REF(name)  _##name##_pooling_gpu_blocked_17718859278101007469_0_0
#define FP64_SUPPORTED 0
#define FP16_SUPPORTED 1
#define FP16_UNIT_USED 1
#define INT8_UNIT_USED 0
#define INT32_UNIT_USED 0
#define INT64_UNIT_USED 0
#define UINT8_UNIT_USED 0
#define UINT32_UNIT_USED 0
#define UNIT_TYPE half
#define UNIT_VAL_MAX HALF_MAX
#define UNIT_VAL_MIN -UNIT_VAL_MAX
#define UNIT_VAL_ONE 1.0h
#define UNIT_VAL_ZERO 0.0h
#define TO_UNIT_TYPE(v) convert_half(v)
#define TO_UNIT_TYPE_SAT(v) convert_half(v)
#define AS_UNIT_TYPE(v) as_half(v)
#define UNIT_MAX_FUNC fmax
#define UNIT_MIN_FUNC fmin
#define UNIT_ABS_FUNC fabs
#define UNIT_TYPE_SIZE 2
#define UNIT_IS_FP 1
#define NL_M as_float(0x0)/*0.000000e+00*/
#define NL_N as_float(0x0)/*0.000000e+00*/
#define ACTIVATION_FUNC_TYPE half
#define ACTIVATION_FUNC_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_VAL_MIN -ACTIVATION_FUNC_VAL_MAX
#define ACTIVATION_FUNC_VAL_ONE 1.0h
#define ACTIVATION_FUNC_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_MAX_FUNC fmax
#define ACTIVATION_FUNC_MIN_FUNC fmin
#define ACTIVATION_FUNC_ABS_FUNC fabs
#define ACTIVATION_FUNC_TYPE_SIZE 2
#define ACTIVATION_FUNC_IS_FP 1
#define ACTIVATION_PARAMS NL_M, NL_N
#define ACTIVATION_FUNC(input, m, n) input
#define ACTIVATION(input, params) ACTIVATION_FUNC(input, params)
#define INPUT0_SIZE_X 112
#define INPUT0_SIZE_Y 112
#define INPUT0_SIZE_Z 1
#define INPUT0_SIZE_W 1
#define INPUT0_SIZE_U 1
#define INPUT0_SIZE_V 1
#define INPUT0_FEATURE_NUM 64
#define INPUT0_BATCH_NUM 1
#define INPUT0_PAD_BEFORE_SIZE_X 0
#define INPUT0_PAD_BEFORE_SIZE_Y 0
#define INPUT0_PAD_BEFORE_SIZE_Z 0
#define INPUT0_PAD_BEFORE_SIZE_W 0
#define INPUT0_PAD_BEFORE_SIZE_U 0
#define INPUT0_PAD_BEFORE_SIZE_V 0
#define INPUT0_PAD_BEFORE_FEATURE_NUM 0
#define INPUT0_PAD_BEFORE_BATCH_NUM 0
#define INPUT0_PAD_AFTER_SIZE_X 0
#define INPUT0_PAD_AFTER_SIZE_Y 0
#define INPUT0_PAD_AFTER_SIZE_Z 0
#define INPUT0_PAD_AFTER_SIZE_W 0
#define INPUT0_PAD_AFTER_SIZE_U 0
#define INPUT0_PAD_AFTER_SIZE_V 0
#define INPUT0_PAD_AFTER_FEATURE_NUM 0
#define INPUT0_PAD_AFTER_BATCH_NUM 0
#define INPUT0_LENGTH 802816
#define INPUT0_X_PITCH 1
#define INPUT0_Y_PITCH 112
#define INPUT0_Z_PITCH 1
#define INPUT0_W_PITCH 1
#define INPUT0_U_PITCH 1
#define INPUT0_V_PITCH 1
#define INPUT0_FEATURE_PITCH 12544
#define INPUT0_BATCH_PITCH 802816
#define INPUT0_GET_INDEX_SAFE(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX_SAFE(INPUT0, b, f, y, x)
#define INPUT0_GET_INDEX(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(INPUT0, b, f, y, x)
#define INPUT0_GET_INDEX_RAW(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(INPUT0, b, f, y, x)
#define INPUT0_VIEW_OFFSET 0
#define INPUT0_DIMS 4
#define INPUT0_SIMPLE 0
#define INPUT0_GROUPED 0
#define INPUT0_LAYOUT_B_FS_YX_FSV16 1
#define INPUT0_TYPE half
#define INPUT0_VAL_MAX HALF_MAX
#define INPUT0_VAL_MIN -INPUT0_VAL_MAX
#define INPUT0_VAL_ONE 1.0h
#define INPUT0_VAL_ZERO 0.0h
#define TO_INPUT0_TYPE(v) convert_half(v)
#define TO_INPUT0_TYPE_SAT(v) convert_half(v)
#define AS_INPUT0_TYPE(v) as_half(v)
#define INPUT0_MAX_FUNC fmax
#define INPUT0_MIN_FUNC fmin
#define INPUT0_ABS_FUNC fabs
#define INPUT0_TYPE_SIZE 2
#define INPUT0_IS_FP 1
#define INPUT0_OFFSET 0
#define INPUT0_SIZES_DATA { 112,112,64,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(INPUT0_SIZES) = INPUT0_SIZES_DATA;
#define INPUT0_SIZES CONST_ARRAY_REF(INPUT0_SIZES)
#define INPUT0_PITCHES (size_t []){ 1,112,12544,802816,1,1,1,1,1, } 
#define INPUT0_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define INPUT0_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OUTPUT_SIZE_X 56
#define OUTPUT_SIZE_Y 56
#define OUTPUT_SIZE_Z 1
#define OUTPUT_SIZE_W 1
#define OUTPUT_SIZE_U 1
#define OUTPUT_SIZE_V 1
#define OUTPUT_FEATURE_NUM 64
#define OUTPUT_BATCH_NUM 1
#define OUTPUT_PAD_BEFORE_SIZE_X 0
#define OUTPUT_PAD_BEFORE_SIZE_Y 0
#define OUTPUT_PAD_BEFORE_SIZE_Z 0
#define OUTPUT_PAD_BEFORE_SIZE_W 0
#define OUTPUT_PAD_BEFORE_SIZE_U 0
#define OUTPUT_PAD_BEFORE_SIZE_V 0
#define OUTPUT_PAD_BEFORE_FEATURE_NUM 0
#define OUTPUT_PAD_BEFORE_BATCH_NUM 0
#define OUTPUT_PAD_AFTER_SIZE_X 0
#define OUTPUT_PAD_AFTER_SIZE_Y 0
#define OUTPUT_PAD_AFTER_SIZE_Z 0
#define OUTPUT_PAD_AFTER_SIZE_W 0
#define OUTPUT_PAD_AFTER_SIZE_U 0
#define OUTPUT_PAD_AFTER_SIZE_V 0
#define OUTPUT_PAD_AFTER_FEATURE_NUM 0
#define OUTPUT_PAD_AFTER_BATCH_NUM 0
#define OUTPUT_LENGTH 200704
#define OUTPUT_X_PITCH 1
#define OUTPUT_Y_PITCH 56
#define OUTPUT_Z_PITCH 1
#define OUTPUT_W_PITCH 1
#define OUTPUT_U_PITCH 1
#define OUTPUT_V_PITCH 1
#define OUTPUT_FEATURE_PITCH 3136
#define OUTPUT_BATCH_PITCH 200704
#define OUTPUT_GET_INDEX_SAFE(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX_SAFE(OUTPUT, b, f, y, x)
#define OUTPUT_GET_INDEX(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(OUTPUT, b, f, y, x)
#define OUTPUT_GET_INDEX_RAW(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(OUTPUT, b, f, y, x)
#define OUTPUT_VIEW_OFFSET 0
#define OUTPUT_DIMS 4
#define OUTPUT_SIMPLE 0
#define OUTPUT_GROUPED 0
#define OUTPUT_LAYOUT_B_FS_YX_FSV16 1
#define OUTPUT_TYPE half
#define OUTPUT_VAL_MAX HALF_MAX
#define OUTPUT_VAL_MIN -OUTPUT_VAL_MAX
#define OUTPUT_VAL_ONE 1.0h
#define OUTPUT_VAL_ZERO 0.0h
#define TO_OUTPUT_TYPE(v) convert_half(v)
#define TO_OUTPUT_TYPE_SAT(v) convert_half(v)
#define AS_OUTPUT_TYPE(v) as_half(v)
#define OUTPUT_MAX_FUNC fmax
#define OUTPUT_MIN_FUNC fmin
#define OUTPUT_ABS_FUNC fabs
#define OUTPUT_TYPE_SIZE 2
#define OUTPUT_IS_FP 1
#define OUTPUT_OFFSET 0
#define OUTPUT_SIZES_DATA { 56,56,64,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(OUTPUT_SIZES) = OUTPUT_SIZES_DATA;
#define OUTPUT_SIZES CONST_ARRAY_REF(OUTPUT_SIZES)
#define OUTPUT_PITCHES (size_t []){ 1,56,3136,200704,1,1,1,1,1, } 
#define OUTPUT_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OUTPUT_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OPTIONAL_SHAPE_INFO_ARG 
#define OPTIONAL_SHAPE_INFO_TENSOR 
#define POOL_SIZE_X 3
#define POOL_SIZE_Y 3
#define POOL_SIZE_Z 1
#define STRIDE_SIZE_X 2
#define STRIDE_SIZE_Y 2
#define STRIDE_SIZE_Z 1
#define PADDING_SIZE_X 1
#define PADDING_SIZE_Y 1
#define PADDING_SIZE_Z 0
#define MAX_POOLING 1
#define FIXED_KERNEL_DIVIDER 1
#define DILATION_SIZE_X 1
#define DILATION_SIZE_Y 1
#define DILATION_SIZE_Z 1
#define CHECK_BOUNDARY 1
#define ACCUMULATOR_TYPE half
#define ACCUMULATOR_VAL_MAX HALF_MAX
#define ACCUMULATOR_VAL_MIN -ACCUMULATOR_VAL_MAX
#define ACCUMULATOR_VAL_ONE 1.0h
#define ACCUMULATOR_VAL_ZERO 0.0h
#define TO_ACCUMULATOR_TYPE(v) convert_half(v)
#define TO_ACCUMULATOR_TYPE_SAT(v) convert_half(v)
#define AS_ACCUMULATOR_TYPE(v) as_half(v)
#define ACCUMULATOR_MAX_FUNC fmax
#define ACCUMULATOR_MIN_FUNC fmin
#define ACCUMULATOR_ABS_FUNC fabs
#define ACCUMULATOR_TYPE_SIZE 2
#define ACCUMULATOR_IS_FP 1
#define CAN_PRELOAD_FULL_LINE 1
#define PADDED_INPUT 0
#define X_BLOCK_SIZE 8
#define INPUT_LINE_SIZE 17
#define SUB_GROUP_SIZE 16
#define X_BLOCKS 7
#define ACTIVATION_TYPE half
#define ACTIVATION_VAL_MAX HALF_MAX
#define ACTIVATION_VAL_MIN -ACTIVATION_VAL_MAX
#define ACTIVATION_VAL_ONE 1.0h
#define ACTIVATION_VAL_ZERO 0.0h
#define TO_ACTIVATION_TYPE(v) convert_half(v)
#define TO_ACTIVATION_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_TYPE(v) as_half(v)
#define ACTIVATION_MAX_FUNC fmax
#define ACTIVATION_MIN_FUNC fmin
#define ACTIVATION_ABS_FUNC fabs
#define ACTIVATION_TYPE_SIZE 2
#define ACTIVATION_IS_FP 1


#define FEATURE_SLICE_SIZE 16
#if X_BLOCK_SIZE > 1
 #define INPUT_VAR_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, X_BLOCK_SIZE)
 #define OUTPUT_VAR_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, X_BLOCK_SIZE)
 #define ACCUMULATOR_VAR_TYPE MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, X_BLOCK_SIZE)
 #define ACTIVATION_VAR_TYPE MAKE_VECTOR_TYPE(ACTIVATION_TYPE, X_BLOCK_SIZE)
#else
 #define INPUT_VAR_TYPE INPUT0_TYPE
 #define OUTPUT_VAR_TYPE OUTPUT_TYPE
 #define ACCUMULATOR_VAR_TYPE ACCUMULATOR_TYPE
 #define ACTIVATION_VAR_TYPE ACTIVATION_TYPE
#endif
#define TO_OUTPUT_VAR_TYPE(x) CAT(convert_, OUTPUT_VAR_TYPE)(x)
#define TO_ACCUMULATOR_VAR_TYPE CAT(convert_, ACCUMULATOR_VAR_TYPE)
#define TO_ACTIVATION_VAR_TYPE CAT(convert_, ACTIVATION_VAR_TYPE)
#if defined MAX_POOLING
 #define INIT_VAL ACCUMULATOR_VAL_MIN
#elif defined AVG_POOLING
 #define INIT_VAL ACCUMULATOR_VAL_ZERO
#endif
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL(pooling_gpu_blocked)(
 const __global INPUT0_TYPE* input,
 __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
 , FUSED_OPS_DECLS
#endif
)
{
 const int lid = get_sub_group_local_id();
#if SUB_GROUP_SIZE == 16
 const int f_block = get_group_id(1);
 const int f_val = 0;
#else
 const int f_block = (uint)get_group_id(1) / (FEATURE_SLICE_SIZE / SUB_GROUP_SIZE);
 const int f_val = (uint)get_group_id(1) % (FEATURE_SLICE_SIZE / SUB_GROUP_SIZE);
#endif
 const int b = get_global_id(2);
 const int xy = get_global_id(0);
 const int x = (xy % X_BLOCKS) * X_BLOCK_SIZE;
 const int y = xy / X_BLOCKS;
 const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
 const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;
 const uint input_x_pitch = FEATURE_SLICE_SIZE;
 const uint input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
 const uint input_fs_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
 const uint input_total_f_size = INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
 const uint input_b_pitch = input_fs_pitch * ((input_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
 const uint input_fs_pad_before = INPUT0_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;
 const uint input_offset = b * input_b_pitch +
 (f_block + input_fs_pad_before) * input_fs_pitch +
 (INPUT0_PAD_BEFORE_SIZE_Y + input_y) * input_y_pitch +
 (INPUT0_PAD_BEFORE_SIZE_X + input_x) * input_x_pitch +
 f_val * SUB_GROUP_SIZE;
 const uint output_x_pitch = FEATURE_SLICE_SIZE;
 const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
 const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
 const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
 const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
 const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;
 const uint output_offset = b * output_b_pitch +
 (f_block + output_fs_pad_before) * output_fs_pitch +
 (y + OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
 (x + OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch +
 f_val * SUB_GROUP_SIZE;
 ACCUMULATOR_VAR_TYPE dst = (ACCUMULATOR_VAR_TYPE)INIT_VAL;
#if AVG_POOLING && (defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER))
 ACCUMULATOR_TYPE count;
 if (lid < X_BLOCK_SIZE)
 {
#if defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
 int y_min = max(-PADDING_SIZE_Y, input_y);
 int x_min = max(-PADDING_SIZE_X, input_x + lid*STRIDE_SIZE_X);
 int x_max = min(input_x + lid*STRIDE_SIZE_X + POOL_SIZE_X, INPUT0_SIZE_X + PADDING_SIZE_X);
 int y_max = min(input_y + POOL_SIZE_Y, INPUT0_SIZE_Y + PADDING_SIZE_Y);
#else
 int y_min = max(0, input_y);
 int x_min = max(0, input_x + lid*STRIDE_SIZE_X);
 int x_max = min(input_x + lid*STRIDE_SIZE_X + POOL_SIZE_X, INPUT0_SIZE_X);
 int y_max = min(input_y + POOL_SIZE_Y, INPUT0_SIZE_Y);
#endif
 count = TO_ACCUMULATOR_TYPE(1.f / (float)((y_max - y_min) * (x_max - x_min)));
 }
 ACCUMULATOR_VAR_TYPE scale;
#if X_BLOCK_SIZE > 1
 for (int i = 0; i < X_BLOCK_SIZE; i++)
 scale[i] = _sub_group_shuffle(count, i);
#else
 scale = _sub_group_shuffle(count, 0);
#endif
#endif
 for (int kh = 0; kh < POOL_SIZE_Y; kh++) {
 if (input_y + kh < 0 || input_y + kh >= INPUT0_SIZE_Y)
 continue;
#if CAN_PRELOAD_FULL_LINE
 INPUT0_TYPE line_cache[INPUT_LINE_SIZE];
 for (int i = 0; i < INPUT_LINE_SIZE; i++) {
 if ((input_x + i) >= 0 && (input_x + i) < INPUT0_SIZE_X)
 line_cache[i] = DT_INPUT_BLOCK_READ(input, input_offset + kh*input_y_pitch + i*input_x_pitch);
 else
 #if defined MAX_POOLING
 line_cache[i] = INPUT0_VAL_MIN;
 #elif defined AVG_POOLING
 line_cache[i] = INPUT0_VAL_ZERO;
 #endif
 }
 __attribute__((opencl_unroll_hint(POOL_SIZE_X)))
 for (int kw = 0; kw < POOL_SIZE_X; kw++)
 {
 ACCUMULATOR_VAR_TYPE src;
#if X_BLOCK_SIZE > 1
 for (int i = 0; i < X_BLOCK_SIZE; i++) {
 src[i] = TO_ACCUMULATOR_TYPE(line_cache[kw + STRIDE_SIZE_X*i]);
 }
#else
 src = TO_ACCUMULATOR_VAR_TYPE(line_cache[kw]);
#endif
#if defined MAX_POOLING
 dst = ACCUMULATOR_MAX_FUNC(dst, src);
#elif defined AVG_POOLING
 dst += src;
#endif
 }
#else
 for (int kw = 0; kw < POOL_SIZE_X; kw++)
 {
 INPUT_VAR_TYPE src;
#if X_BLOCK_SIZE > 1
 for (int i = 0; i < X_BLOCK_SIZE; i++) {
 if ((input_x + kw + STRIDE_SIZE_X*i) >= 0 && (input_x + kw + STRIDE_SIZE_X*i) < INPUT0_SIZE_X)
 src[i] = DT_INPUT_BLOCK_READ(input, input_offset + kh*input_y_pitch + (kw + STRIDE_SIZE_X*i)*input_x_pitch);
 else
 #if defined MAX_POOLING
 src[i] = INPUT0_VAL_MIN;
 #elif defined AVG_POOLING
 src[i] = INPUT0_VAL_ZERO;
 #endif
 }
#else
 src = DT_INPUT_BLOCK_READ(input, input_offset + kh*input_y_pitch + kw*input_x_pitch);
#endif
#if defined MAX_POOLING
 dst = ACCUMULATOR_MAX_FUNC(dst, src);
#elif defined AVG_POOLING
 dst += TO_ACCUMULATOR_VAR_TYPE(src);
#endif
 }
#endif
 }
 ACTIVATION_VAR_TYPE pool_result;
#if defined MAX_POOLING
 pool_result = TO_ACTIVATION_VAR_TYPE(dst);
#elif defined AVG_POOLING && (defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER))
 pool_result = TO_ACTIVATION_VAR_TYPE(dst*scale);
#elif defined AVG_POOLING
 pool_result = TO_ACTIVATION_VAR_TYPE(dst/(POOL_SIZE_X*POOL_SIZE_Y));
#endif
#if !HAS_FUSED_OPS
 pool_result = ACTIVATION(pool_result, ACTIVATION_PARAMS);
#endif
 OUTPUT_VAR_TYPE final_result;
#if OUTPUT_LEFTOVERS
 if (f_block*FEATURE_SLICE_SIZE + (f_val + 1)*SUB_GROUP_SIZE >= OUTPUT_FEATURE_NUM) {
 for (int i = 0; i < X_BLOCK_SIZE; i++) {
 if ((f_block*FEATURE_SLICE_SIZE + f_val*SUB_GROUP_SIZE + lid < OUTPUT_FEATURE_NUM) && (x + i) < OUTPUT_SIZE_X) {
#if X_BLOCK_SIZE > 1
 #if HAS_FUSED_OPS
 FUSED_OPS_SCALAR;
 final_result[i] = FUSED_OPS_RESULT_SCALAR;
 #else
 final_result[i] = TO_OUTPUT_TYPE(pool_result[i]);
 #endif
 output[output_offset + i * output_x_pitch + lid] = final_result[i];
#else
 #if HAS_FUSED_OPS
 FUSED_OPS_VEC;
 final_result = FUSED_OPS_RESULT_VEC;
 #else
 final_result = TO_OUTPUT_VAR_TYPE(pool_result);
 #endif
 output[output_offset + i * output_x_pitch + lid] = final_result;
#endif
 }
 }
 }
 else
#endif
 if (x + X_BLOCK_SIZE <= OUTPUT_SIZE_X)
 {
 #if HAS_FUSED_OPS
 FUSED_OPS_VEC;
 final_result = FUSED_OPS_RESULT_VEC;
 #else
 final_result = TO_OUTPUT_VAR_TYPE(pool_result);
 #endif
#if SUB_GROUP_SIZE == FEATURE_SLICE_SIZE
 #if X_BLOCK_SIZE == 8
 DT_OUTPUT_BLOCK_WRITE8(output, output_offset, final_result);
 #elif X_BLOCK_SIZE == 4
 DT_OUTPUT_BLOCK_WRITE4(output, output_offset, final_result);
 #elif X_BLOCK_SIZE == 2
 DT_OUTPUT_BLOCK_WRITE2(output, output_offset, final_result);
 #elif X_BLOCK_SIZE == 1
 DT_OUTPUT_BLOCK_WRITE(output, output_offset, final_result);
 #endif
#else
 #if X_BLOCK_SIZE > 1
 __attribute__((opencl_unroll_hint(X_BLOCK_SIZE)))
 for (int i = 0; i < X_BLOCK_SIZE; i++) {
 DT_OUTPUT_BLOCK_WRITE(output, output_offset + i * output_x_pitch, final_result[i]);
 }
 #else
 DT_OUTPUT_BLOCK_WRITE(output, output_offset, final_result);
 #endif
#endif
 }
 else
 {
 const int x_tail = OUTPUT_SIZE_X % X_BLOCK_SIZE;
 for (int i = 0; i < x_tail; i++){
#if X_BLOCK_SIZE > 1
 #if HAS_FUSED_OPS
 FUSED_OPS_SCALAR;
 final_result[i] = FUSED_OPS_RESULT_SCALAR;
 #else
 final_result[i] = TO_OUTPUT_TYPE(pool_result[i]);
 #endif
 DT_OUTPUT_BLOCK_WRITE(output, output_offset + i*output_x_pitch, final_result[i]);
#else
 #if HAS_FUSED_OPS
 FUSED_OPS_VEC;
 final_result = FUSED_OPS_RESULT_VEC;
 #else
 final_result = TO_OUTPUT_VAR_TYPE(pool_result);
 #endif
 DT_OUTPUT_BLOCK_WRITE(output, output_offset + i*output_x_pitch, final_result);
#endif
 }
 }
}
#undef INIT_VAL
#undef FEATURE_SLICE_SIZE
#undef INPUT_VAR_TYPE
#undef OUTPUT_VAR_TYPE
#undef TO_OUTPUT_VAR_TYPE
#undef ACCUMULATOR_VAR_TYPE
#undef ACTIVATION_VAR_TYPE
#undef TO_ACTIVATION_VAR_TYPE
#ifdef FEATURE_SLICE_SIZE
#undef FEATURE_SLICE_SIZE
#endif
#ifdef INPUT_VAR_TYPE
#undef INPUT_VAR_TYPE
#endif
#ifdef OUTPUT_VAR_TYPE
#undef OUTPUT_VAR_TYPE
#endif
#ifdef ACCUMULATOR_VAR_TYPE
#undef ACCUMULATOR_VAR_TYPE
#endif
#ifdef ACTIVATION_VAR_TYPE
#undef ACTIVATION_VAR_TYPE
#endif
#ifdef INPUT_VAR_TYPE
#undef INPUT_VAR_TYPE
#endif
#ifdef OUTPUT_VAR_TYPE
#undef OUTPUT_VAR_TYPE
#endif
#ifdef ACCUMULATOR_VAR_TYPE
#undef ACCUMULATOR_VAR_TYPE
#endif
#ifdef ACTIVATION_VAR_TYPE
#undef ACTIVATION_VAR_TYPE
#endif
#ifdef TO_OUTPUT_VAR_TYPE
#undef TO_OUTPUT_VAR_TYPE
#endif
#ifdef TO_ACCUMULATOR_VAR_TYPE
#undef TO_ACCUMULATOR_VAR_TYPE
#endif
#ifdef TO_ACTIVATION_VAR_TYPE
#undef TO_ACTIVATION_VAR_TYPE
#endif
#ifdef INIT_VAL
#undef INIT_VAL
#endif
#ifdef INIT_VAL
#undef INIT_VAL
#endif
#undef KERNEL
#undef KERNEL_ID
#undef FUNC
#undef FUNC_CALL
#undef CONST_ARRAY_DECL
#undef CONST_ARRAY_REF
#ifdef FP64_SUPPORTED
#undef FP64_SUPPORTED
#endif
#ifdef FP16_SUPPORTED
#undef FP16_SUPPORTED
#endif
#ifdef FP16_UNIT_USED
#undef FP16_UNIT_USED
#endif
#ifdef INT8_UNIT_USED
#undef INT8_UNIT_USED
#endif
#ifdef INT32_UNIT_USED
#undef INT32_UNIT_USED
#endif
#ifdef INT64_UNIT_USED
#undef INT64_UNIT_USED
#endif
#ifdef UINT8_UNIT_USED
#undef UINT8_UNIT_USED
#endif
#ifdef UINT32_UNIT_USED
#undef UINT32_UNIT_USED
#endif
#ifdef UNIT_TYPE
#undef UNIT_TYPE
#endif
#ifdef UNIT_VAL_MAX
#undef UNIT_VAL_MAX
#endif
#ifdef UNIT_VAL_MIN
#undef UNIT_VAL_MIN
#endif
#ifdef UNIT_VAL_ONE
#undef UNIT_VAL_ONE
#endif
#ifdef UNIT_VAL_ZERO
#undef UNIT_VAL_ZERO
#endif
#ifdef TO_UNIT_TYPE
#undef TO_UNIT_TYPE
#endif
#ifdef TO_UNIT_TYPE_SAT
#undef TO_UNIT_TYPE_SAT
#endif
#ifdef AS_UNIT_TYPE
#undef AS_UNIT_TYPE
#endif
#ifdef UNIT_MAX_FUNC
#undef UNIT_MAX_FUNC
#endif
#ifdef UNIT_MIN_FUNC
#undef UNIT_MIN_FUNC
#endif
#ifdef UNIT_ABS_FUNC
#undef UNIT_ABS_FUNC
#endif
#ifdef UNIT_TYPE_SIZE
#undef UNIT_TYPE_SIZE
#endif
#ifdef UNIT_IS_FP
#undef UNIT_IS_FP
#endif
#ifdef NL_M
#undef NL_M
#endif
#ifdef NL_N
#undef NL_N
#endif
#ifdef ACTIVATION_FUNC_TYPE
#undef ACTIVATION_FUNC_TYPE
#endif
#ifdef ACTIVATION_FUNC_VAL_MAX
#undef ACTIVATION_FUNC_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_VAL_MIN
#undef ACTIVATION_FUNC_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_VAL_ONE
#undef ACTIVATION_FUNC_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_VAL_ZERO
#undef ACTIVATION_FUNC_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_TYPE
#undef TO_ACTIVATION_FUNC_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_TYPE_SAT
#undef TO_ACTIVATION_FUNC_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_TYPE
#undef AS_ACTIVATION_FUNC_TYPE
#endif
#ifdef ACTIVATION_FUNC_MAX_FUNC
#undef ACTIVATION_FUNC_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_MIN_FUNC
#undef ACTIVATION_FUNC_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_ABS_FUNC
#undef ACTIVATION_FUNC_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_TYPE_SIZE
#undef ACTIVATION_FUNC_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_IS_FP
#undef ACTIVATION_FUNC_IS_FP
#endif
#ifdef ACTIVATION_PARAMS
#undef ACTIVATION_PARAMS
#endif
#ifdef ACTIVATION_FUNC
#undef ACTIVATION_FUNC
#endif
#ifdef ACTIVATION
#undef ACTIVATION
#endif
#ifdef INPUT0_SIZE_X
#undef INPUT0_SIZE_X
#endif
#ifdef INPUT0_SIZE_Y
#undef INPUT0_SIZE_Y
#endif
#ifdef INPUT0_SIZE_Z
#undef INPUT0_SIZE_Z
#endif
#ifdef INPUT0_SIZE_W
#undef INPUT0_SIZE_W
#endif
#ifdef INPUT0_SIZE_U
#undef INPUT0_SIZE_U
#endif
#ifdef INPUT0_SIZE_V
#undef INPUT0_SIZE_V
#endif
#ifdef INPUT0_FEATURE_NUM
#undef INPUT0_FEATURE_NUM
#endif
#ifdef INPUT0_BATCH_NUM
#undef INPUT0_BATCH_NUM
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_X
#undef INPUT0_PAD_BEFORE_SIZE_X
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_Y
#undef INPUT0_PAD_BEFORE_SIZE_Y
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_Z
#undef INPUT0_PAD_BEFORE_SIZE_Z
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_W
#undef INPUT0_PAD_BEFORE_SIZE_W
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_U
#undef INPUT0_PAD_BEFORE_SIZE_U
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_V
#undef INPUT0_PAD_BEFORE_SIZE_V
#endif
#ifdef INPUT0_PAD_BEFORE_FEATURE_NUM
#undef INPUT0_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef INPUT0_PAD_BEFORE_BATCH_NUM
#undef INPUT0_PAD_BEFORE_BATCH_NUM
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_X
#undef INPUT0_PAD_AFTER_SIZE_X
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_Y
#undef INPUT0_PAD_AFTER_SIZE_Y
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_Z
#undef INPUT0_PAD_AFTER_SIZE_Z
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_W
#undef INPUT0_PAD_AFTER_SIZE_W
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_U
#undef INPUT0_PAD_AFTER_SIZE_U
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_V
#undef INPUT0_PAD_AFTER_SIZE_V
#endif
#ifdef INPUT0_PAD_AFTER_FEATURE_NUM
#undef INPUT0_PAD_AFTER_FEATURE_NUM
#endif
#ifdef INPUT0_PAD_AFTER_BATCH_NUM
#undef INPUT0_PAD_AFTER_BATCH_NUM
#endif
#ifdef INPUT0_LENGTH
#undef INPUT0_LENGTH
#endif
#ifdef INPUT0_X_PITCH
#undef INPUT0_X_PITCH
#endif
#ifdef INPUT0_Y_PITCH
#undef INPUT0_Y_PITCH
#endif
#ifdef INPUT0_Z_PITCH
#undef INPUT0_Z_PITCH
#endif
#ifdef INPUT0_W_PITCH
#undef INPUT0_W_PITCH
#endif
#ifdef INPUT0_U_PITCH
#undef INPUT0_U_PITCH
#endif
#ifdef INPUT0_V_PITCH
#undef INPUT0_V_PITCH
#endif
#ifdef INPUT0_FEATURE_PITCH
#undef INPUT0_FEATURE_PITCH
#endif
#ifdef INPUT0_BATCH_PITCH
#undef INPUT0_BATCH_PITCH
#endif
#ifdef INPUT0_GET_INDEX_SAFE
#undef INPUT0_GET_INDEX_SAFE
#endif
#ifdef INPUT0_GET_INDEX
#undef INPUT0_GET_INDEX
#endif
#ifdef INPUT0_GET_INDEX_RAW
#undef INPUT0_GET_INDEX_RAW
#endif
#ifdef INPUT0_VIEW_OFFSET
#undef INPUT0_VIEW_OFFSET
#endif
#ifdef INPUT0_DIMS
#undef INPUT0_DIMS
#endif
#ifdef INPUT0_SIMPLE
#undef INPUT0_SIMPLE
#endif
#ifdef INPUT0_GROUPED
#undef INPUT0_GROUPED
#endif
#ifdef INPUT0_LAYOUT_B_FS_YX_FSV16
#undef INPUT0_LAYOUT_B_FS_YX_FSV16
#endif
#ifdef INPUT0_TYPE
#undef INPUT0_TYPE
#endif
#ifdef INPUT0_VAL_MAX
#undef INPUT0_VAL_MAX
#endif
#ifdef INPUT0_VAL_MIN
#undef INPUT0_VAL_MIN
#endif
#ifdef INPUT0_VAL_ONE
#undef INPUT0_VAL_ONE
#endif
#ifdef INPUT0_VAL_ZERO
#undef INPUT0_VAL_ZERO
#endif
#ifdef TO_INPUT0_TYPE
#undef TO_INPUT0_TYPE
#endif
#ifdef TO_INPUT0_TYPE_SAT
#undef TO_INPUT0_TYPE_SAT
#endif
#ifdef AS_INPUT0_TYPE
#undef AS_INPUT0_TYPE
#endif
#ifdef INPUT0_MAX_FUNC
#undef INPUT0_MAX_FUNC
#endif
#ifdef INPUT0_MIN_FUNC
#undef INPUT0_MIN_FUNC
#endif
#ifdef INPUT0_ABS_FUNC
#undef INPUT0_ABS_FUNC
#endif
#ifdef INPUT0_TYPE_SIZE
#undef INPUT0_TYPE_SIZE
#endif
#ifdef INPUT0_IS_FP
#undef INPUT0_IS_FP
#endif
#ifdef INPUT0_OFFSET
#undef INPUT0_OFFSET
#endif
#ifdef INPUT0_SIZES_DATA
#undef INPUT0_SIZES_DATA
#endif
#ifdef INPUT0_PITCHES
#undef INPUT0_PITCHES
#endif
#ifdef INPUT0_PAD_BEFORE
#undef INPUT0_PAD_BEFORE
#endif
#ifdef INPUT0_PAD_AFTER
#undef INPUT0_PAD_AFTER
#endif
#ifdef OUTPUT_SIZE_X
#undef OUTPUT_SIZE_X
#endif
#ifdef OUTPUT_SIZE_Y
#undef OUTPUT_SIZE_Y
#endif
#ifdef OUTPUT_SIZE_Z
#undef OUTPUT_SIZE_Z
#endif
#ifdef OUTPUT_SIZE_W
#undef OUTPUT_SIZE_W
#endif
#ifdef OUTPUT_SIZE_U
#undef OUTPUT_SIZE_U
#endif
#ifdef OUTPUT_SIZE_V
#undef OUTPUT_SIZE_V
#endif
#ifdef OUTPUT_FEATURE_NUM
#undef OUTPUT_FEATURE_NUM
#endif
#ifdef OUTPUT_BATCH_NUM
#undef OUTPUT_BATCH_NUM
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_X
#undef OUTPUT_PAD_BEFORE_SIZE_X
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_Y
#undef OUTPUT_PAD_BEFORE_SIZE_Y
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_Z
#undef OUTPUT_PAD_BEFORE_SIZE_Z
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_W
#undef OUTPUT_PAD_BEFORE_SIZE_W
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_U
#undef OUTPUT_PAD_BEFORE_SIZE_U
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_V
#undef OUTPUT_PAD_BEFORE_SIZE_V
#endif
#ifdef OUTPUT_PAD_BEFORE_FEATURE_NUM
#undef OUTPUT_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef OUTPUT_PAD_BEFORE_BATCH_NUM
#undef OUTPUT_PAD_BEFORE_BATCH_NUM
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_X
#undef OUTPUT_PAD_AFTER_SIZE_X
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_Y
#undef OUTPUT_PAD_AFTER_SIZE_Y
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_Z
#undef OUTPUT_PAD_AFTER_SIZE_Z
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_W
#undef OUTPUT_PAD_AFTER_SIZE_W
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_U
#undef OUTPUT_PAD_AFTER_SIZE_U
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_V
#undef OUTPUT_PAD_AFTER_SIZE_V
#endif
#ifdef OUTPUT_PAD_AFTER_FEATURE_NUM
#undef OUTPUT_PAD_AFTER_FEATURE_NUM
#endif
#ifdef OUTPUT_PAD_AFTER_BATCH_NUM
#undef OUTPUT_PAD_AFTER_BATCH_NUM
#endif
#ifdef OUTPUT_LENGTH
#undef OUTPUT_LENGTH
#endif
#ifdef OUTPUT_X_PITCH
#undef OUTPUT_X_PITCH
#endif
#ifdef OUTPUT_Y_PITCH
#undef OUTPUT_Y_PITCH
#endif
#ifdef OUTPUT_Z_PITCH
#undef OUTPUT_Z_PITCH
#endif
#ifdef OUTPUT_W_PITCH
#undef OUTPUT_W_PITCH
#endif
#ifdef OUTPUT_U_PITCH
#undef OUTPUT_U_PITCH
#endif
#ifdef OUTPUT_V_PITCH
#undef OUTPUT_V_PITCH
#endif
#ifdef OUTPUT_FEATURE_PITCH
#undef OUTPUT_FEATURE_PITCH
#endif
#ifdef OUTPUT_BATCH_PITCH
#undef OUTPUT_BATCH_PITCH
#endif
#ifdef OUTPUT_GET_INDEX_SAFE
#undef OUTPUT_GET_INDEX_SAFE
#endif
#ifdef OUTPUT_GET_INDEX
#undef OUTPUT_GET_INDEX
#endif
#ifdef OUTPUT_GET_INDEX_RAW
#undef OUTPUT_GET_INDEX_RAW
#endif
#ifdef OUTPUT_VIEW_OFFSET
#undef OUTPUT_VIEW_OFFSET
#endif
#ifdef OUTPUT_DIMS
#undef OUTPUT_DIMS
#endif
#ifdef OUTPUT_SIMPLE
#undef OUTPUT_SIMPLE
#endif
#ifdef OUTPUT_GROUPED
#undef OUTPUT_GROUPED
#endif
#ifdef OUTPUT_LAYOUT_B_FS_YX_FSV16
#undef OUTPUT_LAYOUT_B_FS_YX_FSV16
#endif
#ifdef OUTPUT_TYPE
#undef OUTPUT_TYPE
#endif
#ifdef OUTPUT_VAL_MAX
#undef OUTPUT_VAL_MAX
#endif
#ifdef OUTPUT_VAL_MIN
#undef OUTPUT_VAL_MIN
#endif
#ifdef OUTPUT_VAL_ONE
#undef OUTPUT_VAL_ONE
#endif
#ifdef OUTPUT_VAL_ZERO
#undef OUTPUT_VAL_ZERO
#endif
#ifdef TO_OUTPUT_TYPE
#undef TO_OUTPUT_TYPE
#endif
#ifdef TO_OUTPUT_TYPE_SAT
#undef TO_OUTPUT_TYPE_SAT
#endif
#ifdef AS_OUTPUT_TYPE
#undef AS_OUTPUT_TYPE
#endif
#ifdef OUTPUT_MAX_FUNC
#undef OUTPUT_MAX_FUNC
#endif
#ifdef OUTPUT_MIN_FUNC
#undef OUTPUT_MIN_FUNC
#endif
#ifdef OUTPUT_ABS_FUNC
#undef OUTPUT_ABS_FUNC
#endif
#ifdef OUTPUT_TYPE_SIZE
#undef OUTPUT_TYPE_SIZE
#endif
#ifdef OUTPUT_IS_FP
#undef OUTPUT_IS_FP
#endif
#ifdef OUTPUT_OFFSET
#undef OUTPUT_OFFSET
#endif
#ifdef OUTPUT_SIZES_DATA
#undef OUTPUT_SIZES_DATA
#endif
#ifdef OUTPUT_PITCHES
#undef OUTPUT_PITCHES
#endif
#ifdef OUTPUT_PAD_BEFORE
#undef OUTPUT_PAD_BEFORE
#endif
#ifdef OUTPUT_PAD_AFTER
#undef OUTPUT_PAD_AFTER
#endif
#ifdef OPTIONAL_SHAPE_INFO_ARG
#undef OPTIONAL_SHAPE_INFO_ARG
#endif
#ifdef OPTIONAL_SHAPE_INFO_TENSOR
#undef OPTIONAL_SHAPE_INFO_TENSOR
#endif
#ifdef POOL_SIZE_X
#undef POOL_SIZE_X
#endif
#ifdef POOL_SIZE_Y
#undef POOL_SIZE_Y
#endif
#ifdef POOL_SIZE_Z
#undef POOL_SIZE_Z
#endif
#ifdef STRIDE_SIZE_X
#undef STRIDE_SIZE_X
#endif
#ifdef STRIDE_SIZE_Y
#undef STRIDE_SIZE_Y
#endif
#ifdef STRIDE_SIZE_Z
#undef STRIDE_SIZE_Z
#endif
#ifdef PADDING_SIZE_X
#undef PADDING_SIZE_X
#endif
#ifdef PADDING_SIZE_Y
#undef PADDING_SIZE_Y
#endif
#ifdef PADDING_SIZE_Z
#undef PADDING_SIZE_Z
#endif
#ifdef MAX_POOLING
#undef MAX_POOLING
#endif
#ifdef FIXED_KERNEL_DIVIDER
#undef FIXED_KERNEL_DIVIDER
#endif
#ifdef DILATION_SIZE_X
#undef DILATION_SIZE_X
#endif
#ifdef DILATION_SIZE_Y
#undef DILATION_SIZE_Y
#endif
#ifdef DILATION_SIZE_Z
#undef DILATION_SIZE_Z
#endif
#ifdef CHECK_BOUNDARY
#undef CHECK_BOUNDARY
#endif
#ifdef ACCUMULATOR_TYPE
#undef ACCUMULATOR_TYPE
#endif
#ifdef ACCUMULATOR_VAL_MAX
#undef ACCUMULATOR_VAL_MAX
#endif
#ifdef ACCUMULATOR_VAL_MIN
#undef ACCUMULATOR_VAL_MIN
#endif
#ifdef ACCUMULATOR_VAL_ONE
#undef ACCUMULATOR_VAL_ONE
#endif
#ifdef ACCUMULATOR_VAL_ZERO
#undef ACCUMULATOR_VAL_ZERO
#endif
#ifdef TO_ACCUMULATOR_TYPE
#undef TO_ACCUMULATOR_TYPE
#endif
#ifdef TO_ACCUMULATOR_TYPE_SAT
#undef TO_ACCUMULATOR_TYPE_SAT
#endif
#ifdef AS_ACCUMULATOR_TYPE
#undef AS_ACCUMULATOR_TYPE
#endif
#ifdef ACCUMULATOR_MAX_FUNC
#undef ACCUMULATOR_MAX_FUNC
#endif
#ifdef ACCUMULATOR_MIN_FUNC
#undef ACCUMULATOR_MIN_FUNC
#endif
#ifdef ACCUMULATOR_ABS_FUNC
#undef ACCUMULATOR_ABS_FUNC
#endif
#ifdef ACCUMULATOR_TYPE_SIZE
#undef ACCUMULATOR_TYPE_SIZE
#endif
#ifdef ACCUMULATOR_IS_FP
#undef ACCUMULATOR_IS_FP
#endif
#ifdef CAN_PRELOAD_FULL_LINE
#undef CAN_PRELOAD_FULL_LINE
#endif
#ifdef PADDED_INPUT
#undef PADDED_INPUT
#endif
#ifdef X_BLOCK_SIZE
#undef X_BLOCK_SIZE
#endif
#ifdef INPUT_LINE_SIZE
#undef INPUT_LINE_SIZE
#endif
#ifdef SUB_GROUP_SIZE
#undef SUB_GROUP_SIZE
#endif
#ifdef X_BLOCKS
#undef X_BLOCKS
#endif
#ifdef ACTIVATION_TYPE
#undef ACTIVATION_TYPE
#endif
#ifdef ACTIVATION_VAL_MAX
#undef ACTIVATION_VAL_MAX
#endif
#ifdef ACTIVATION_VAL_MIN
#undef ACTIVATION_VAL_MIN
#endif
#ifdef ACTIVATION_VAL_ONE
#undef ACTIVATION_VAL_ONE
#endif
#ifdef ACTIVATION_VAL_ZERO
#undef ACTIVATION_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_TYPE
#undef TO_ACTIVATION_TYPE
#endif
#ifdef TO_ACTIVATION_TYPE_SAT
#undef TO_ACTIVATION_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_TYPE
#undef AS_ACTIVATION_TYPE
#endif
#ifdef ACTIVATION_MAX_FUNC
#undef ACTIVATION_MAX_FUNC
#endif
#ifdef ACTIVATION_MIN_FUNC
#undef ACTIVATION_MIN_FUNC
#endif
#ifdef ACTIVATION_ABS_FUNC
#undef ACTIVATION_ABS_FUNC
#endif
#ifdef ACTIVATION_TYPE_SIZE
#undef ACTIVATION_TYPE_SIZE
#endif
#ifdef ACTIVATION_IS_FP
#undef ACTIVATION_IS_FP
#endif

//====================================================
// Kernel template: reorder_data 
// Kernel name: reorder_data_11475626742907933301_0_0
#define KERNEL(name) __kernel void reorder_data_11475626742907933301_0_0
#define KERNEL_ID reorder_data_11475626742907933301_0_0
#define FUNC(name)  _##name##_reorder_data_11475626742907933301_0_0
#define FUNC_CALL(name)  _##name##_reorder_data_11475626742907933301_0_0
#define CONST_ARRAY_DECL(name) __constant size_t  _##name##_reorder_data_11475626742907933301_0_0 []
#define CONST_ARRAY_REF(name)  _##name##_reorder_data_11475626742907933301_0_0
#define FP64_SUPPORTED 0
#define FP16_SUPPORTED 1
#define FP16_UNIT_USED 1
#define INT8_UNIT_USED 0
#define INT32_UNIT_USED 0
#define INT64_UNIT_USED 0
#define UINT8_UNIT_USED 1
#define UINT32_UNIT_USED 0
#define UNIT_TYPE half
#define UNIT_VAL_MAX HALF_MAX
#define UNIT_VAL_MIN -UNIT_VAL_MAX
#define UNIT_VAL_ONE 1.0h
#define UNIT_VAL_ZERO 0.0h
#define TO_UNIT_TYPE(v) convert_half(v)
#define TO_UNIT_TYPE_SAT(v) convert_half(v)
#define AS_UNIT_TYPE(v) as_half(v)
#define UNIT_MAX_FUNC fmax
#define UNIT_MIN_FUNC fmin
#define UNIT_ABS_FUNC fabs
#define UNIT_TYPE_SIZE 2
#define UNIT_IS_FP 1
#define NL_M as_float(0x0)/*0.000000e+00*/
#define NL_N as_float(0x0)/*0.000000e+00*/
#define ACTIVATION_FUNC_TYPE half
#define ACTIVATION_FUNC_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_VAL_MIN -ACTIVATION_FUNC_VAL_MAX
#define ACTIVATION_FUNC_VAL_ONE 1.0h
#define ACTIVATION_FUNC_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_MAX_FUNC fmax
#define ACTIVATION_FUNC_MIN_FUNC fmin
#define ACTIVATION_FUNC_ABS_FUNC fabs
#define ACTIVATION_FUNC_TYPE_SIZE 2
#define ACTIVATION_FUNC_IS_FP 1
#define ACTIVATION_PARAMS NL_M, NL_N
#define ACTIVATION_FUNC(input, m, n) input
#define ACTIVATION(input, params) ACTIVATION_FUNC(input, params)
#define INPUT0_SIZE_X 224
#define INPUT0_SIZE_Y 224
#define INPUT0_SIZE_Z 1
#define INPUT0_SIZE_W 1
#define INPUT0_SIZE_U 1
#define INPUT0_SIZE_V 1
#define INPUT0_FEATURE_NUM 3
#define INPUT0_BATCH_NUM 1
#define INPUT0_PAD_BEFORE_SIZE_X 0
#define INPUT0_PAD_BEFORE_SIZE_Y 0
#define INPUT0_PAD_BEFORE_SIZE_Z 0
#define INPUT0_PAD_BEFORE_SIZE_W 0
#define INPUT0_PAD_BEFORE_SIZE_U 0
#define INPUT0_PAD_BEFORE_SIZE_V 0
#define INPUT0_PAD_BEFORE_FEATURE_NUM 0
#define INPUT0_PAD_BEFORE_BATCH_NUM 0
#define INPUT0_PAD_AFTER_SIZE_X 0
#define INPUT0_PAD_AFTER_SIZE_Y 0
#define INPUT0_PAD_AFTER_SIZE_Z 0
#define INPUT0_PAD_AFTER_SIZE_W 0
#define INPUT0_PAD_AFTER_SIZE_U 0
#define INPUT0_PAD_AFTER_SIZE_V 0
#define INPUT0_PAD_AFTER_FEATURE_NUM 0
#define INPUT0_PAD_AFTER_BATCH_NUM 0
#define INPUT0_LENGTH 150528
#define INPUT0_X_PITCH 1
#define INPUT0_Y_PITCH 224
#define INPUT0_Z_PITCH 1
#define INPUT0_W_PITCH 1
#define INPUT0_U_PITCH 1
#define INPUT0_V_PITCH 1
#define INPUT0_FEATURE_PITCH 50176
#define INPUT0_BATCH_PITCH 150528
#define INPUT0_GET_INDEX_SAFE(b, f, y, x) GET_DATA_INDEX_SAFE(INPUT0, b, f, y, x)
#define INPUT0_GET_INDEX(b, f, y, x) GET_DATA_INDEX(INPUT0, b, f, y, x)
#define INPUT0_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(INPUT0, b, f, y, x)
#define INPUT0_VIEW_OFFSET 0
#define INPUT0_DIMS 4
#define INPUT0_SIMPLE 1
#define INPUT0_GROUPED 0
#define INPUT0_LAYOUT_BFYX 1
#define INPUT0_TYPE uchar
#define INPUT0_VAL_MAX UCHAR_MAX
#define INPUT0_VAL_MIN 0
#define INPUT0_VAL_ONE (uchar) 1
#define INPUT0_VAL_ZERO (uchar) 0
#define TO_INPUT0_TYPE(v) convert_uchar(v)
#define TO_INPUT0_TYPE_SAT(v) convert_uchar_sat(v)
#define AS_INPUT0_TYPE(v) as_uchar(v)
#define INPUT0_MAX_FUNC max
#define INPUT0_MIN_FUNC min
#define INPUT0_ABS_FUNC abs
#define INPUT0_TYPE_SIZE 1
#define INPUT0_IS_FP 0
#define INPUT0_OFFSET 0
#define INPUT0_SIZES_DATA { 224,224,3,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(INPUT0_SIZES) = INPUT0_SIZES_DATA;
#define INPUT0_SIZES CONST_ARRAY_REF(INPUT0_SIZES)
#define INPUT0_PITCHES (size_t []){ 1,224,50176,150528,1,1,1,1,1, } 
#define INPUT0_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define INPUT0_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OUTPUT_SIZE_X 224
#define OUTPUT_SIZE_Y 224
#define OUTPUT_SIZE_Z 1
#define OUTPUT_SIZE_W 1
#define OUTPUT_SIZE_U 1
#define OUTPUT_SIZE_V 1
#define OUTPUT_FEATURE_NUM 3
#define OUTPUT_BATCH_NUM 1
#define OUTPUT_PAD_BEFORE_SIZE_X 0
#define OUTPUT_PAD_BEFORE_SIZE_Y 0
#define OUTPUT_PAD_BEFORE_SIZE_Z 0
#define OUTPUT_PAD_BEFORE_SIZE_W 0
#define OUTPUT_PAD_BEFORE_SIZE_U 0
#define OUTPUT_PAD_BEFORE_SIZE_V 0
#define OUTPUT_PAD_BEFORE_FEATURE_NUM 0
#define OUTPUT_PAD_BEFORE_BATCH_NUM 0
#define OUTPUT_PAD_AFTER_SIZE_X 0
#define OUTPUT_PAD_AFTER_SIZE_Y 0
#define OUTPUT_PAD_AFTER_SIZE_Z 0
#define OUTPUT_PAD_AFTER_SIZE_W 0
#define OUTPUT_PAD_AFTER_SIZE_U 0
#define OUTPUT_PAD_AFTER_SIZE_V 0
#define OUTPUT_PAD_AFTER_FEATURE_NUM 0
#define OUTPUT_PAD_AFTER_BATCH_NUM 0
#define OUTPUT_LENGTH 150528
#define OUTPUT_X_PITCH 1
#define OUTPUT_Y_PITCH 224
#define OUTPUT_Z_PITCH 1
#define OUTPUT_W_PITCH 1
#define OUTPUT_U_PITCH 1
#define OUTPUT_V_PITCH 1
#define OUTPUT_FEATURE_PITCH 50176
#define OUTPUT_BATCH_PITCH 150528
#define OUTPUT_GET_INDEX_SAFE(b, f, y, x) GET_DATA_INDEX_SAFE(OUTPUT, b, f, y, x)
#define OUTPUT_GET_INDEX(b, f, y, x) GET_DATA_INDEX(OUTPUT, b, f, y, x)
#define OUTPUT_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(OUTPUT, b, f, y, x)
#define OUTPUT_VIEW_OFFSET 0
#define OUTPUT_DIMS 4
#define OUTPUT_SIMPLE 1
#define OUTPUT_GROUPED 0
#define OUTPUT_LAYOUT_BFYX 1
#define OUTPUT_TYPE half
#define OUTPUT_VAL_MAX HALF_MAX
#define OUTPUT_VAL_MIN -OUTPUT_VAL_MAX
#define OUTPUT_VAL_ONE 1.0h
#define OUTPUT_VAL_ZERO 0.0h
#define TO_OUTPUT_TYPE(v) convert_half(v)
#define TO_OUTPUT_TYPE_SAT(v) convert_half(v)
#define AS_OUTPUT_TYPE(v) as_half(v)
#define OUTPUT_MAX_FUNC fmax
#define OUTPUT_MIN_FUNC fmin
#define OUTPUT_ABS_FUNC fabs
#define OUTPUT_TYPE_SIZE 2
#define OUTPUT_IS_FP 1
#define OUTPUT_OFFSET 0
#define OUTPUT_SIZES_DATA { 224,224,3,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(OUTPUT_SIZES) = OUTPUT_SIZES_DATA;
#define OUTPUT_SIZES CONST_ARRAY_REF(OUTPUT_SIZES)
#define OUTPUT_PITCHES (size_t []){ 1,224,50176,150528,1,1,1,1,1, } 
#define OUTPUT_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OUTPUT_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OPTIONAL_SHAPE_INFO_ARG 
#define OPTIONAL_SHAPE_INFO_TENSOR 
#define MEAN_SUBTRACT_NONE 1
#define CALC_TYPE uchar
#define CALC_VAL_MAX UCHAR_MAX
#define CALC_VAL_MIN 0
#define CALC_VAL_ONE (uchar) 1
#define CALC_VAL_ZERO (uchar) 0
#define TO_CALC_TYPE(v) convert_uchar(v)
#define TO_CALC_TYPE_SAT(v) convert_uchar_sat(v)
#define AS_CALC_TYPE(v) as_uchar(v)
#define CALC_MAX_FUNC max
#define CALC_MIN_FUNC min
#define CALC_ABS_FUNC abs
#define CALC_TYPE_SIZE 1
#define CALC_IS_FP 0
#define INPUT_REORDER_TYPE uchar
#define INPUT_REORDER_VAL_MAX UCHAR_MAX
#define INPUT_REORDER_VAL_MIN 0
#define INPUT_REORDER_VAL_ONE (uchar) 1
#define INPUT_REORDER_VAL_ZERO (uchar) 0
#define TO_INPUT_REORDER_TYPE(v) convert_uchar(v)
#define TO_INPUT_REORDER_TYPE_SAT(v) convert_uchar_sat(v)
#define AS_INPUT_REORDER_TYPE(v) as_uchar(v)
#define INPUT_REORDER_MAX_FUNC max
#define INPUT_REORDER_MIN_FUNC min
#define INPUT_REORDER_ABS_FUNC abs
#define INPUT_REORDER_TYPE_SIZE 1
#define INPUT_REORDER_IS_FP 0
#define OUTPUT_REORDER_TYPE half
#define OUTPUT_REORDER_VAL_MAX HALF_MAX
#define OUTPUT_REORDER_VAL_MIN -OUTPUT_REORDER_VAL_MAX
#define OUTPUT_REORDER_VAL_ONE 1.0h
#define OUTPUT_REORDER_VAL_ZERO 0.0h
#define TO_OUTPUT_REORDER_TYPE(v) convert_half(v)
#define TO_OUTPUT_REORDER_TYPE_SAT(v) convert_half(v)
#define AS_OUTPUT_REORDER_TYPE(v) as_half(v)
#define OUTPUT_REORDER_MAX_FUNC fmax
#define OUTPUT_REORDER_MIN_FUNC fmin
#define OUTPUT_REORDER_ABS_FUNC fabs
#define OUTPUT_REORDER_TYPE_SIZE 2
#define OUTPUT_REORDER_IS_FP 1
#define MEAN_OP(val, mean_val) val-mean_val
#define NL_M_TYPED as_float(0x0)/*0.000000e+00*/
#define NL_N_TYPED as_float(0x0)/*0.000000e+00*/
#define ACTIVATION_FUNC_TYPED_TYPE half
#define ACTIVATION_FUNC_TYPED_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_TYPED_VAL_MIN -ACTIVATION_FUNC_TYPED_VAL_MAX
#define ACTIVATION_FUNC_TYPED_VAL_ONE 1.0h
#define ACTIVATION_FUNC_TYPED_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_TYPED_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_TYPED_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_TYPED_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_TYPED_MAX_FUNC fmax
#define ACTIVATION_FUNC_TYPED_MIN_FUNC fmin
#define ACTIVATION_FUNC_TYPED_ABS_FUNC fabs
#define ACTIVATION_FUNC_TYPED_TYPE_SIZE 2
#define ACTIVATION_FUNC_TYPED_IS_FP 1
#define ACTIVATION_PARAMS_TYPED NL_M_TYPED, NL_N_TYPED
#define ACTIVATION_FUNC_TYPED(jit_type, input, m, n) input
#define ACTIVATION_TYPED(jit_type, input, params) ACTIVATION_FUNC_TYPED(jit_type, input, params)
#define SUB_GROUP_SIZE 1
#define CONVERT_TRUNCATE 1
#define GWS_BATCH 2
#define GWS_FEATURE 1
#define GWS_YX 0


inline uint8 FUNC(reshape_dims)(
 uint b, uint f, uint v, uint u, uint w, uint z, uint y, uint x,
 uint src_size_f, uint src_size_v, uint src_size_u, uint src_size_w, uint src_size_z, uint src_size_y, uint src_size_x,
 uint dst_size_f, uint dst_size_v, uint dst_size_u, uint dst_size_w, uint dst_size_z, uint dst_size_y, uint dst_size_x,
 uint src_dims, uint dst_dims)
{
 if (dst_dims == src_dims) {
 return (uint8)(b, f, v, u, w, z, y, x);
 }
 const uint src_pitch_x = 1;
 const uint src_pitch_y = src_pitch_x * src_size_x;
 const uint src_pitch_z = src_pitch_y * src_size_y;
 const uint src_pitch_w = src_pitch_z * src_size_z;
 const uint src_pitch_u = src_pitch_w * src_size_w;
 const uint src_pitch_v = src_pitch_u * src_size_u;
 const uint src_pitch_f = src_pitch_v * src_size_v;
 const uint src_pitch_b = src_pitch_f * src_size_f;
 uint flat_idx = x * src_pitch_x
 + y * src_pitch_y
 + z * src_pitch_z
 + w * src_pitch_w
 + u * src_pitch_u
 + v * src_pitch_v
 + f * src_pitch_f
 + b * src_pitch_b;
 uint dst_x = flat_idx % dst_size_x;
 flat_idx /= dst_size_x;
 uint dst_y = flat_idx % dst_size_y;
 flat_idx /= dst_size_y;
 uint dst_z = flat_idx % dst_size_z;
 flat_idx /= dst_size_z;
 uint dst_w = flat_idx % dst_size_w;
 flat_idx /= dst_size_w;
 uint dst_u = flat_idx % dst_size_u;
 flat_idx /= dst_size_u;
 uint dst_v = flat_idx % dst_size_v;
 flat_idx /= dst_size_v;
 uint dst_f = flat_idx % dst_size_f;
 flat_idx /= dst_size_f;
 uint dst_b = flat_idx;
 return (uint8)(dst_b, dst_f, dst_v, dst_u, dst_w, dst_z, dst_y, dst_x);
}
#define RESHAPE_DIMS(src_prefix, dst_prefix, b, f, v, u, w, z, y, x) \
 FUNC_CALL(reshape_dims)( \
 b, f, v, u, w, z, y, x, \
 CAT(src_prefix, _FEATURE_NUM), \
 CAT(src_prefix, _SIZE_V), \
 CAT(src_prefix, _SIZE_U), \
 CAT(src_prefix, _SIZE_W), \
 CAT(src_prefix, _SIZE_Z), \
 CAT(src_prefix, _SIZE_Y), \
 CAT(src_prefix, _SIZE_X), \
 CAT(dst_prefix, _FEATURE_NUM), \
 CAT(dst_prefix, _SIZE_V), \
 CAT(dst_prefix, _SIZE_U), \
 CAT(dst_prefix, _SIZE_W), \
 CAT(dst_prefix, _SIZE_Z), \
 CAT(dst_prefix, _SIZE_Y), \
 CAT(dst_prefix, _SIZE_X), \
 CAT(src_prefix, _DIMS), \
 CAT(dst_prefix, _DIMS))
inline uint FUNC(get_input_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint v, uint u, uint w, uint z, uint y, uint x) __attribute__((overloadable)) {
#if INPUT0_DIMS < 5
 return INPUT0_GET_INDEX(b, f, y, x);
#elif INPUT0_DIMS == 5
 return INPUT0_GET_INDEX(b, f, z, y, x);
#elif INPUT0_DIMS == 6
 return INPUT0_GET_INDEX(b, f, w, z, y, x);
#elif INPUT0_DIMS == 7
 return INPUT0_GET_INDEX(b, f, u, w, z, y, x);
#elif INPUT0_DIMS == 8
 return INPUT0_GET_INDEX(b, f, v, u, w, z, y, x);
#else
#error [GPU] Unsupported input tensor rank in get_input_index function
#endif
}
inline uint FUNC(get_input_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) __attribute__((overloadable)) {
 return FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, 0, 0, w, z, y, x);
}
inline uint FUNC(get_output_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint v, uint u, uint w, uint z, uint y, uint x) __attribute__((overloadable)) {
#if OUTPUT_DIMS < 5
 return OUTPUT_GET_INDEX(b, f, y, x);
#elif OUTPUT_DIMS == 5
 return OUTPUT_GET_INDEX(b, f, z, y, x);
#elif OUTPUT_DIMS == 6
 return OUTPUT_GET_INDEX(b, f, w, z, y, x);
#elif OUTPUT_DIMS == 7
 return OUTPUT_GET_INDEX(b, f, u, w, z, y, x);
#elif OUTPUT_DIMS == 8
 return OUTPUT_GET_INDEX(b, f, v, u, w, z, y, x);
#else
#error [GPU] Unsupported output tensor rank in get_output_index function
#endif
}
inline uint FUNC(get_output_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) __attribute__((overloadable)) {
 return FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, 0, 0, w, z, y, x);
}
#define DECLARE_SAMPLER const sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST
#if FP16_UNIT_USED
 #define IMAGE_READ(image, coord) read_imageh((image), imageSampler, (coord))
 #define IMAGE_WRITE(image, coord, val) write_imageh((image), (coord), (val))
#else
 #define IMAGE_READ(image, coord) read_imagef((image), imageSampler, (coord))
 #define IMAGE_WRITE(image, coord, val) write_imagef((image), (coord), (val))
#endif
#define INPUT_TYPE4 MAKE_VECTOR_TYPE(INPUT_REORDER_TYPE, 4)
#define OUTPUT_TYPE4 MAKE_VECTOR_TYPE(OUTPUT_REORDER_TYPE, 4)
KERNEL (reorder_data)(
 OPTIONAL_SHAPE_INFO_ARG
#if INPUT0_LAYOUT_NV12 || INPUT0_LAYOUT_IMAGE_2D_RGBA || SURFACE_INPUT
 read_only image2d_t input,
#else
 const __global INPUT_REORDER_TYPE* input,
#endif
#if OUTPUT_LAYOUT_IMAGE_2D_RGBA
 write_only image2d_t output
#else
 __global OUTPUT_REORDER_TYPE* output
#endif
#ifdef MEAN_SUBTRACT_IN_BUFFER
 , __global MEAN_SUBTRACT_TYPE* mean_subtract
#endif
 )
{
#if INPUT0_LAYOUT_BYFX
 const uint b = get_global_id(GWS_BATCH);
 const uint y = get_global_id(GWS_FEATURE);
#else
 const uint b = get_global_id(GWS_BATCH);
 const uint f = get_global_id(GWS_FEATURE);
#endif
#if INPUT0_DIMS == 2
 const uint y = 0;
 const uint x = 0;
 const uint z = 0;
 const uint w = 0;
 const uint u = 0;
 const uint v = 0;
#elif INPUT0_DIMS == 4
 #if INPUT0_LAYOUT_BYFX
 const uint f = ((uint)(get_global_id(GWS_YX))) / INPUT0_SIZE_X;
 const uint x = ((uint)(get_global_id(GWS_YX))) % INPUT0_SIZE_X;
 #else
 const uint y = ((uint)(get_global_id(GWS_YX))) / INPUT0_SIZE_X;
 const uint x = ((uint)(get_global_id(GWS_YX))) % INPUT0_SIZE_X;
 #endif
 const uint z = 0;
 const uint w = 0;
 const uint u = 0;
 const uint v = 0;
#elif INPUT0_DIMS == 5
 uint data_idx = get_global_id(GWS_YX);
 uint tmp_data_idx = data_idx / INPUT0_SIZE_X;
 const uint x = data_idx - tmp_data_idx * INPUT0_SIZE_X;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / INPUT0_SIZE_Y;
 const uint y = data_idx - tmp_data_idx * INPUT0_SIZE_Y;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / INPUT0_SIZE_Z;
 const uint z = data_idx - tmp_data_idx * INPUT0_SIZE_Z;
 const uint w = 0;
 const uint u = 0;
 const uint v = 0;
#elif INPUT0_DIMS == 6
 const uint gid_yx = (uint)(get_global_id(GWS_YX));
 const uint x = gid_yx % INPUT0_SIZE_X;
 const uint y = gid_yx / INPUT0_SIZE_X % INPUT0_SIZE_Y;
 const uint z = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
 const uint w = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y / INPUT0_SIZE_Z % INPUT0_SIZE_W;
 const uint u = 0;
 const uint v = 0;
#elif INPUT0_DIMS == 7
 const uint gid_yx = (uint)(get_global_id(GWS_YX));
 const uint x = gid_yx % INPUT0_SIZE_X;
 const uint y = gid_yx / INPUT0_SIZE_X % INPUT0_SIZE_Y;
 const uint z = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
 const uint w = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y / INPUT0_SIZE_Z % INPUT0_SIZE_W;
 const uint u = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y / INPUT0_SIZE_Z / INPUT0_SIZE_W % INPUT0_SIZE_U;
 const uint v = 0;
#elif INPUT0_DIMS == 8
 const uint gid_yx = (uint)(get_global_id(GWS_YX));
 const uint x = gid_yx % INPUT0_SIZE_X;
 const uint y = gid_yx / INPUT0_SIZE_X % INPUT0_SIZE_Y;
 const uint z = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
 const uint w = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y / INPUT0_SIZE_Z % INPUT0_SIZE_W;
 const uint u = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y / INPUT0_SIZE_Z / INPUT0_SIZE_W % INPUT0_SIZE_U;
 const uint v = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y / INPUT0_SIZE_Z / INPUT0_SIZE_W / INPUT0_SIZE_U % INPUT0_SIZE_V;
#endif
#if defined INPUT0_LAYOUT_NV12 && !SURFACE_INPUT
 const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;
 float4 colorVYU = read_imagef(input, sampler, (int2)(y, f));
 float Ycomponent = mad(colorVYU.s1, 296.82f, -18.624f);
 float Ucomponent = mad(colorVYU.s2, 255.0f, -128.f);
 float Vcomponent = mad(colorVYU.s0, 255.0f, -128.f);
 float B = clamp(mad(Vcomponent, 1.596f, Ycomponent), 0.f, 255.f);
 float R = clamp(mad(Ucomponent, 2.018f, Ycomponent), 0.f, 255.f);
 float G = clamp(mad(Vcomponent, -0.813f, mad(Ucomponent, -0.391f, Ycomponent)), 0.f, 255.f);
#elif defined INPUT0_LAYOUT_IMAGE_2D_RGBA
 const sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;
 OUTPUT_TYPE4 colorRGBA = IMAGE_READ(input, (int2)(x, y));
#elif defined OUTPUT_LAYOUT_IMAGE_2D_RGBA
 uint8 ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, f, v, u, w, z, y, x);
 const uint input_idx_R = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, 0, v, u, w, z, y, x);
 const uint input_idx_G = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, 1, v, u, w, z, y, x);
 const uint input_idx_B = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, 2, v, u, w, z, y, x);
#if OUTPUT_FEATURE_NUM == 3
 INPUT_TYPE4 colorRGBA = { TO_INPUT_REORDER_TYPE(input[input_idx_R]), TO_INPUT_REORDER_TYPE(input[input_idx_G]), TO_INPUT_REORDER_TYPE(input[input_idx_B]), TO_INPUT_REORDER_TYPE(0.f) };
#else
 const uint input_idx_A = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, 3, v, u, w, z, y, x);
 INPUT_TYPE4 colorRGBA = { TO_INPUT_REORDER_TYPE(input[input_idx_R]), TO_INPUT_REORDER_TYPE(input[input_idx_G]), TO_INPUT_REORDER_TYPE(input[input_idx_B]), TO_INPUT_REORDER_TYPE(input[input_idx_A]) };
#endif
#else
 uint8 ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, f, v, u, w, z, y, x);
 const uint input_idx = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, v, u, w, z, y, x);
 const uint output_idx = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR ov.s0, ov.s1, ov.s2, ov.s3, ov.s4, ov.s5, ov.s6, ov.s7);
#if defined MEAN_SUBTRACT_INSIDE_PARAMS
 float res = TO_MEAN_TYPE(input[input_idx]);
 res = MEAN_OP(res, VALUE_TO_SUBTRACT[f % VALUE_TO_SUBTRACT_SIZE]);
#elif defined MEAN_SUBTRACT_IN_BUFFER
#if defined MEAN_PER_FEATURE
 MEAN_SUBTRACT_TYPE res = TO_MEAN_TYPE(input[input_idx]);
 res = MEAN_OP(res, mean_subtract[f]);
#else
 MEAN_SUBTRACT_TYPE res = TO_MEAN_TYPE(input[input_idx]);
 uint8 msv = RESHAPE_DIMS(INPUT0, MEAN_SUBTRACT, b, f, v, u, w, z, y, x);
 res = MEAN_OP(res, mean_subtract[GET_DATA_INDEX_SAFE(MEAN_SUBTRACT, msv.s0, msv.s1, msv.s6, msv.s7)]);
#endif
#elif SURFACE_INPUT
 float4 Y = read_imagef(input, (int2)(y, f));
 float Ycomponent = mad(Y.x, 296.82f, -18.624f);
 float res = clamp(Ycomponent, 0.f, 255.f);
#else
 #ifdef BF16_INPUT
 CALC_TYPE res = TO_CALC_TYPE(_convert_as_bfloat16_float(input[input_idx]));
 #elif defined INT4_INPUT
 const uint uint4_idx = input_idx >> 1;
 OUTPUT_TYPE res = TO_OUTPUT_REORDER_TYPE(convert_as_int4_float(input[uint4_idx], input_idx));
 #elif defined UINT4_INPUT
 const uint uint4_idx = input_idx >> 1;
 OUTPUT_TYPE res = TO_OUTPUT_REORDER_TYPE(convert_as_uint4_float(input[uint4_idx], input_idx));
 #else
 CALC_TYPE res = TO_CALC_TYPE(input[input_idx]);
 #endif
#endif
#endif
#if defined INPUT0_LAYOUT_NV12 && !SURFACE_INPUT
 uint8 ov0 = RESHAPE_DIMS(INPUT0, OUTPUT, b, 0, v, u, w, z, y, x);
 uint8 ov1 = RESHAPE_DIMS(INPUT0, OUTPUT, b, 1, v, u, w, z, y, x);
 uint8 ov2 = RESHAPE_DIMS(INPUT0, OUTPUT, b, 2, v, u, w, z, y, x);
 uint output_idx_R = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR ov0.s0, ov0.s1, ov0.s2, ov0.s3, ov0.s4, ov0.s5, ov0.s6, ov0.s7);
 uint output_idx_G = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR ov1.s0, ov1.s1, ov1.s2, ov1.s3, ov1.s4, ov1.s5, ov1.s6, ov1.s7);
 uint output_idx_B = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR ov2.s0, ov2.s1, ov2.s2, ov2.s3, ov2.s4, ov2.s5, ov2.s6, ov2.s7);
 #if HAS_FUSED_OPS
 res = TO_OUTPUT_REORDER_TYPE(R);
 FUSED_OPS;
 output[output_idx_R] = FUSED_OPS_RESULT;
 res = TO_OUTPUT_REORDER_TYPE(G);
 FUSED_OPS;
 output[output_idx_G] = FUSED_OPS_RESULT;
 res = TO_OUTPUT_REORDER_TYPE(B);
 FUSED_OPS;
 output[output_idx_B] = FUSED_OPS_RESULT;
 #else
 output[output_idx_R] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(R), NL_M, NL_N);
 output[output_idx_G] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(G), NL_M, NL_N);
 output[output_idx_B] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(B), NL_M, NL_N);
 #endif
#elif INPUT0_LAYOUT_IMAGE_2D_RGBA
 uint8 ov0 = RESHAPE_DIMS(INPUT0, OUTPUT, b, 0, v, u, w, z, y, x);
 uint8 ov1 = RESHAPE_DIMS(INPUT0, OUTPUT, b, 1, v, u, w, z, y, x);
 uint8 ov2 = RESHAPE_DIMS(INPUT0, OUTPUT, b, 2, v, u, w, z, y, x);
 uint output_idx_0 = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR ov0.s0, ov0.s1, ov0.s2, ov0.s3, ov0.s4, ov0.s5, ov0.s6, ov0.s7);
 uint output_idx_1 = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR ov1.s0, ov1.s1, ov1.s2, ov1.s3, ov1.s4, ov1.s5, ov1.s6, ov1.s7);
 uint output_idx_2 = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR ov2.s0, ov2.s1, ov2.s2, ov2.s3, ov2.s4, ov2.s5, ov2.s6, ov2.s7);
 #if HAS_FUSED_OPS
 res = TO_OUTPUT_REORDER_TYPE(colorRGBA.s0);
 FUSED_OPS;
 output[output_idx_0] = FUSED_OPS_RESULT;
 res = TO_OUTPUT_REORDER_TYPE(colorRGBA.s1);
 FUSED_OPS;
 output[output_idx_1] = FUSED_OPS_RESULT;
 res = TO_OUTPUT_REORDER_TYPE(colorRGBA.s2);
 FUSED_OPS;
 output[output_idx_2] = FUSED_OPS_RESULT;
 #else
 output[output_idx_0] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(colorRGBA.s0), NL_M, NL_N);
 output[output_idx_1] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(colorRGBA.s1), NL_M, NL_N);
 output[output_idx_2] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(colorRGBA.s2), NL_M, NL_N);
 #endif
 #if INPUT0_FEATURE_NUM == 4
 uint8 ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 3, v, u, w, z, y, x);
 uint output_idx = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR ov.s0, ov.s1, ov.s2, ov.s3, ov.s4, ov.s5, ov.s6, ov.s7);
 #if HAS_FUSED_OPS
 res = TO_OUTPUT_REORDER_TYPE(colorRGBA.s3);
 FUSED_OPS;
 output[output_idx] = FUSED_OPS_RESULT;
 #else
 output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(colorRGBA.s3), NL_M, NL_N);
 #endif
 #endif
#elif OUTPUT_LAYOUT_IMAGE_2D_RGBA
 IMAGE_WRITE(output, (int2)(x, y), colorRGBA);
#else
 #if INPUT0_IS_FP && !OUTPUT_IS_FP
 #if CONVERT_TRUNCATE
 #define __TO_OUTPUT_REORDER_TYPE(res) TO_OUTPUT_REORDER_TYPE(convert_long(res))
 #else
 #define __TO_OUTPUT_REORDER_TYPE(res) TO_OUTPUT_REORDER_TYPE_SAT(res)
 #endif
 #else
 #define __TO_OUTPUT_REORDER_TYPE(res) TO_OUTPUT_REORDER_TYPE(res)
 #endif
 #if HAS_FUSED_OPS
 res = __TO_OUTPUT_REORDER_TYPE(res);
 FUSED_OPS;
 output[output_idx] = FUSED_OPS_RESULT;
 #elif defined(INT4_OUTPUT) || defined(UINT4_OUTPUT)
 OUTPUT_TYPE val_char = __TO_OUTPUT_REORDER_TYPE(res);
 int val_i32 = convert_int(val_char);
 #if !CONVERT_TRUNCATE
 #if defined(INT4_OUTPUT)
 val_i32 = clamp(val_i32, -8, 7);
 #else
 val_i32 = clamp(val_i32, 0, 15);
 #endif
 #endif
 uint val_u32 = (uint)(val_i32 & 0x0F);
 volatile __global uint* output_u32 = (volatile __global uint*)output;
 uint main_idx = output_idx / 8;
 uint sub_idx = output_idx % 8;
 uint shift = sub_idx * 4;
 atomic_and(&output_u32[main_idx], ~(0x0F << shift));
 atomic_or(&output_u32[main_idx], (val_u32 << shift));
 #else
 output[output_idx] = ACTIVATION_TYPED(OUTPUT_REORDER, __TO_OUTPUT_REORDER_TYPE(res), ACTIVATION_PARAMS_TYPED);
 #endif
#undef __TO_OUTPUT_REORDER_TYPE
#endif
}
#undef INPUT_TYPE4
#undef OUTPUT_TYPE4
#ifdef INPUT_TYPE4
#undef INPUT_TYPE4
#endif
#ifdef OUTPUT_TYPE4
#undef OUTPUT_TYPE4
#endif
#ifdef __TO_OUTPUT_REORDER_TYPE
#undef __TO_OUTPUT_REORDER_TYPE
#endif
#ifdef __TO_OUTPUT_REORDER_TYPE
#undef __TO_OUTPUT_REORDER_TYPE
#endif
#ifdef __TO_OUTPUT_REORDER_TYPE
#undef __TO_OUTPUT_REORDER_TYPE
#endif
#ifdef RESHAPE_DIMS
#undef RESHAPE_DIMS
#endif
#ifdef DECLARE_SAMPLER
#undef DECLARE_SAMPLER
#endif
#ifdef IMAGE_READ
#undef IMAGE_READ
#endif
#ifdef IMAGE_WRITE
#undef IMAGE_WRITE
#endif
#ifdef IMAGE_READ
#undef IMAGE_READ
#endif
#ifdef IMAGE_WRITE
#undef IMAGE_WRITE
#endif
#undef KERNEL
#undef KERNEL_ID
#undef FUNC
#undef FUNC_CALL
#undef CONST_ARRAY_DECL
#undef CONST_ARRAY_REF
#ifdef FP64_SUPPORTED
#undef FP64_SUPPORTED
#endif
#ifdef FP16_SUPPORTED
#undef FP16_SUPPORTED
#endif
#ifdef FP16_UNIT_USED
#undef FP16_UNIT_USED
#endif
#ifdef INT8_UNIT_USED
#undef INT8_UNIT_USED
#endif
#ifdef INT32_UNIT_USED
#undef INT32_UNIT_USED
#endif
#ifdef INT64_UNIT_USED
#undef INT64_UNIT_USED
#endif
#ifdef UINT8_UNIT_USED
#undef UINT8_UNIT_USED
#endif
#ifdef UINT32_UNIT_USED
#undef UINT32_UNIT_USED
#endif
#ifdef UNIT_TYPE
#undef UNIT_TYPE
#endif
#ifdef UNIT_VAL_MAX
#undef UNIT_VAL_MAX
#endif
#ifdef UNIT_VAL_MIN
#undef UNIT_VAL_MIN
#endif
#ifdef UNIT_VAL_ONE
#undef UNIT_VAL_ONE
#endif
#ifdef UNIT_VAL_ZERO
#undef UNIT_VAL_ZERO
#endif
#ifdef TO_UNIT_TYPE
#undef TO_UNIT_TYPE
#endif
#ifdef TO_UNIT_TYPE_SAT
#undef TO_UNIT_TYPE_SAT
#endif
#ifdef AS_UNIT_TYPE
#undef AS_UNIT_TYPE
#endif
#ifdef UNIT_MAX_FUNC
#undef UNIT_MAX_FUNC
#endif
#ifdef UNIT_MIN_FUNC
#undef UNIT_MIN_FUNC
#endif
#ifdef UNIT_ABS_FUNC
#undef UNIT_ABS_FUNC
#endif
#ifdef UNIT_TYPE_SIZE
#undef UNIT_TYPE_SIZE
#endif
#ifdef UNIT_IS_FP
#undef UNIT_IS_FP
#endif
#ifdef NL_M
#undef NL_M
#endif
#ifdef NL_N
#undef NL_N
#endif
#ifdef ACTIVATION_FUNC_TYPE
#undef ACTIVATION_FUNC_TYPE
#endif
#ifdef ACTIVATION_FUNC_VAL_MAX
#undef ACTIVATION_FUNC_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_VAL_MIN
#undef ACTIVATION_FUNC_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_VAL_ONE
#undef ACTIVATION_FUNC_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_VAL_ZERO
#undef ACTIVATION_FUNC_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_TYPE
#undef TO_ACTIVATION_FUNC_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_TYPE_SAT
#undef TO_ACTIVATION_FUNC_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_TYPE
#undef AS_ACTIVATION_FUNC_TYPE
#endif
#ifdef ACTIVATION_FUNC_MAX_FUNC
#undef ACTIVATION_FUNC_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_MIN_FUNC
#undef ACTIVATION_FUNC_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_ABS_FUNC
#undef ACTIVATION_FUNC_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_TYPE_SIZE
#undef ACTIVATION_FUNC_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_IS_FP
#undef ACTIVATION_FUNC_IS_FP
#endif
#ifdef ACTIVATION_PARAMS
#undef ACTIVATION_PARAMS
#endif
#ifdef ACTIVATION_FUNC
#undef ACTIVATION_FUNC
#endif
#ifdef ACTIVATION
#undef ACTIVATION
#endif
#ifdef INPUT0_SIZE_X
#undef INPUT0_SIZE_X
#endif
#ifdef INPUT0_SIZE_Y
#undef INPUT0_SIZE_Y
#endif
#ifdef INPUT0_SIZE_Z
#undef INPUT0_SIZE_Z
#endif
#ifdef INPUT0_SIZE_W
#undef INPUT0_SIZE_W
#endif
#ifdef INPUT0_SIZE_U
#undef INPUT0_SIZE_U
#endif
#ifdef INPUT0_SIZE_V
#undef INPUT0_SIZE_V
#endif
#ifdef INPUT0_FEATURE_NUM
#undef INPUT0_FEATURE_NUM
#endif
#ifdef INPUT0_BATCH_NUM
#undef INPUT0_BATCH_NUM
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_X
#undef INPUT0_PAD_BEFORE_SIZE_X
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_Y
#undef INPUT0_PAD_BEFORE_SIZE_Y
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_Z
#undef INPUT0_PAD_BEFORE_SIZE_Z
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_W
#undef INPUT0_PAD_BEFORE_SIZE_W
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_U
#undef INPUT0_PAD_BEFORE_SIZE_U
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_V
#undef INPUT0_PAD_BEFORE_SIZE_V
#endif
#ifdef INPUT0_PAD_BEFORE_FEATURE_NUM
#undef INPUT0_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef INPUT0_PAD_BEFORE_BATCH_NUM
#undef INPUT0_PAD_BEFORE_BATCH_NUM
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_X
#undef INPUT0_PAD_AFTER_SIZE_X
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_Y
#undef INPUT0_PAD_AFTER_SIZE_Y
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_Z
#undef INPUT0_PAD_AFTER_SIZE_Z
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_W
#undef INPUT0_PAD_AFTER_SIZE_W
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_U
#undef INPUT0_PAD_AFTER_SIZE_U
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_V
#undef INPUT0_PAD_AFTER_SIZE_V
#endif
#ifdef INPUT0_PAD_AFTER_FEATURE_NUM
#undef INPUT0_PAD_AFTER_FEATURE_NUM
#endif
#ifdef INPUT0_PAD_AFTER_BATCH_NUM
#undef INPUT0_PAD_AFTER_BATCH_NUM
#endif
#ifdef INPUT0_LENGTH
#undef INPUT0_LENGTH
#endif
#ifdef INPUT0_X_PITCH
#undef INPUT0_X_PITCH
#endif
#ifdef INPUT0_Y_PITCH
#undef INPUT0_Y_PITCH
#endif
#ifdef INPUT0_Z_PITCH
#undef INPUT0_Z_PITCH
#endif
#ifdef INPUT0_W_PITCH
#undef INPUT0_W_PITCH
#endif
#ifdef INPUT0_U_PITCH
#undef INPUT0_U_PITCH
#endif
#ifdef INPUT0_V_PITCH
#undef INPUT0_V_PITCH
#endif
#ifdef INPUT0_FEATURE_PITCH
#undef INPUT0_FEATURE_PITCH
#endif
#ifdef INPUT0_BATCH_PITCH
#undef INPUT0_BATCH_PITCH
#endif
#ifdef INPUT0_GET_INDEX_SAFE
#undef INPUT0_GET_INDEX_SAFE
#endif
#ifdef INPUT0_GET_INDEX
#undef INPUT0_GET_INDEX
#endif
#ifdef INPUT0_GET_INDEX_RAW
#undef INPUT0_GET_INDEX_RAW
#endif
#ifdef INPUT0_VIEW_OFFSET
#undef INPUT0_VIEW_OFFSET
#endif
#ifdef INPUT0_DIMS
#undef INPUT0_DIMS
#endif
#ifdef INPUT0_SIMPLE
#undef INPUT0_SIMPLE
#endif
#ifdef INPUT0_GROUPED
#undef INPUT0_GROUPED
#endif
#ifdef INPUT0_LAYOUT_BFYX
#undef INPUT0_LAYOUT_BFYX
#endif
#ifdef INPUT0_TYPE
#undef INPUT0_TYPE
#endif
#ifdef INPUT0_VAL_MAX
#undef INPUT0_VAL_MAX
#endif
#ifdef INPUT0_VAL_MIN
#undef INPUT0_VAL_MIN
#endif
#ifdef INPUT0_VAL_ONE
#undef INPUT0_VAL_ONE
#endif
#ifdef INPUT0_VAL_ZERO
#undef INPUT0_VAL_ZERO
#endif
#ifdef TO_INPUT0_TYPE
#undef TO_INPUT0_TYPE
#endif
#ifdef TO_INPUT0_TYPE_SAT
#undef TO_INPUT0_TYPE_SAT
#endif
#ifdef AS_INPUT0_TYPE
#undef AS_INPUT0_TYPE
#endif
#ifdef INPUT0_MAX_FUNC
#undef INPUT0_MAX_FUNC
#endif
#ifdef INPUT0_MIN_FUNC
#undef INPUT0_MIN_FUNC
#endif
#ifdef INPUT0_ABS_FUNC
#undef INPUT0_ABS_FUNC
#endif
#ifdef INPUT0_TYPE_SIZE
#undef INPUT0_TYPE_SIZE
#endif
#ifdef INPUT0_IS_FP
#undef INPUT0_IS_FP
#endif
#ifdef INPUT0_OFFSET
#undef INPUT0_OFFSET
#endif
#ifdef INPUT0_SIZES_DATA
#undef INPUT0_SIZES_DATA
#endif
#ifdef INPUT0_PITCHES
#undef INPUT0_PITCHES
#endif
#ifdef INPUT0_PAD_BEFORE
#undef INPUT0_PAD_BEFORE
#endif
#ifdef INPUT0_PAD_AFTER
#undef INPUT0_PAD_AFTER
#endif
#ifdef OUTPUT_SIZE_X
#undef OUTPUT_SIZE_X
#endif
#ifdef OUTPUT_SIZE_Y
#undef OUTPUT_SIZE_Y
#endif
#ifdef OUTPUT_SIZE_Z
#undef OUTPUT_SIZE_Z
#endif
#ifdef OUTPUT_SIZE_W
#undef OUTPUT_SIZE_W
#endif
#ifdef OUTPUT_SIZE_U
#undef OUTPUT_SIZE_U
#endif
#ifdef OUTPUT_SIZE_V
#undef OUTPUT_SIZE_V
#endif
#ifdef OUTPUT_FEATURE_NUM
#undef OUTPUT_FEATURE_NUM
#endif
#ifdef OUTPUT_BATCH_NUM
#undef OUTPUT_BATCH_NUM
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_X
#undef OUTPUT_PAD_BEFORE_SIZE_X
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_Y
#undef OUTPUT_PAD_BEFORE_SIZE_Y
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_Z
#undef OUTPUT_PAD_BEFORE_SIZE_Z
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_W
#undef OUTPUT_PAD_BEFORE_SIZE_W
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_U
#undef OUTPUT_PAD_BEFORE_SIZE_U
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_V
#undef OUTPUT_PAD_BEFORE_SIZE_V
#endif
#ifdef OUTPUT_PAD_BEFORE_FEATURE_NUM
#undef OUTPUT_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef OUTPUT_PAD_BEFORE_BATCH_NUM
#undef OUTPUT_PAD_BEFORE_BATCH_NUM
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_X
#undef OUTPUT_PAD_AFTER_SIZE_X
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_Y
#undef OUTPUT_PAD_AFTER_SIZE_Y
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_Z
#undef OUTPUT_PAD_AFTER_SIZE_Z
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_W
#undef OUTPUT_PAD_AFTER_SIZE_W
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_U
#undef OUTPUT_PAD_AFTER_SIZE_U
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_V
#undef OUTPUT_PAD_AFTER_SIZE_V
#endif
#ifdef OUTPUT_PAD_AFTER_FEATURE_NUM
#undef OUTPUT_PAD_AFTER_FEATURE_NUM
#endif
#ifdef OUTPUT_PAD_AFTER_BATCH_NUM
#undef OUTPUT_PAD_AFTER_BATCH_NUM
#endif
#ifdef OUTPUT_LENGTH
#undef OUTPUT_LENGTH
#endif
#ifdef OUTPUT_X_PITCH
#undef OUTPUT_X_PITCH
#endif
#ifdef OUTPUT_Y_PITCH
#undef OUTPUT_Y_PITCH
#endif
#ifdef OUTPUT_Z_PITCH
#undef OUTPUT_Z_PITCH
#endif
#ifdef OUTPUT_W_PITCH
#undef OUTPUT_W_PITCH
#endif
#ifdef OUTPUT_U_PITCH
#undef OUTPUT_U_PITCH
#endif
#ifdef OUTPUT_V_PITCH
#undef OUTPUT_V_PITCH
#endif
#ifdef OUTPUT_FEATURE_PITCH
#undef OUTPUT_FEATURE_PITCH
#endif
#ifdef OUTPUT_BATCH_PITCH
#undef OUTPUT_BATCH_PITCH
#endif
#ifdef OUTPUT_GET_INDEX_SAFE
#undef OUTPUT_GET_INDEX_SAFE
#endif
#ifdef OUTPUT_GET_INDEX
#undef OUTPUT_GET_INDEX
#endif
#ifdef OUTPUT_GET_INDEX_RAW
#undef OUTPUT_GET_INDEX_RAW
#endif
#ifdef OUTPUT_VIEW_OFFSET
#undef OUTPUT_VIEW_OFFSET
#endif
#ifdef OUTPUT_DIMS
#undef OUTPUT_DIMS
#endif
#ifdef OUTPUT_SIMPLE
#undef OUTPUT_SIMPLE
#endif
#ifdef OUTPUT_GROUPED
#undef OUTPUT_GROUPED
#endif
#ifdef OUTPUT_LAYOUT_BFYX
#undef OUTPUT_LAYOUT_BFYX
#endif
#ifdef OUTPUT_TYPE
#undef OUTPUT_TYPE
#endif
#ifdef OUTPUT_VAL_MAX
#undef OUTPUT_VAL_MAX
#endif
#ifdef OUTPUT_VAL_MIN
#undef OUTPUT_VAL_MIN
#endif
#ifdef OUTPUT_VAL_ONE
#undef OUTPUT_VAL_ONE
#endif
#ifdef OUTPUT_VAL_ZERO
#undef OUTPUT_VAL_ZERO
#endif
#ifdef TO_OUTPUT_TYPE
#undef TO_OUTPUT_TYPE
#endif
#ifdef TO_OUTPUT_TYPE_SAT
#undef TO_OUTPUT_TYPE_SAT
#endif
#ifdef AS_OUTPUT_TYPE
#undef AS_OUTPUT_TYPE
#endif
#ifdef OUTPUT_MAX_FUNC
#undef OUTPUT_MAX_FUNC
#endif
#ifdef OUTPUT_MIN_FUNC
#undef OUTPUT_MIN_FUNC
#endif
#ifdef OUTPUT_ABS_FUNC
#undef OUTPUT_ABS_FUNC
#endif
#ifdef OUTPUT_TYPE_SIZE
#undef OUTPUT_TYPE_SIZE
#endif
#ifdef OUTPUT_IS_FP
#undef OUTPUT_IS_FP
#endif
#ifdef OUTPUT_OFFSET
#undef OUTPUT_OFFSET
#endif
#ifdef OUTPUT_SIZES_DATA
#undef OUTPUT_SIZES_DATA
#endif
#ifdef OUTPUT_PITCHES
#undef OUTPUT_PITCHES
#endif
#ifdef OUTPUT_PAD_BEFORE
#undef OUTPUT_PAD_BEFORE
#endif
#ifdef OUTPUT_PAD_AFTER
#undef OUTPUT_PAD_AFTER
#endif
#ifdef OPTIONAL_SHAPE_INFO_ARG
#undef OPTIONAL_SHAPE_INFO_ARG
#endif
#ifdef OPTIONAL_SHAPE_INFO_TENSOR
#undef OPTIONAL_SHAPE_INFO_TENSOR
#endif
#ifdef MEAN_SUBTRACT_NONE
#undef MEAN_SUBTRACT_NONE
#endif
#ifdef CALC_TYPE
#undef CALC_TYPE
#endif
#ifdef CALC_VAL_MAX
#undef CALC_VAL_MAX
#endif
#ifdef CALC_VAL_MIN
#undef CALC_VAL_MIN
#endif
#ifdef CALC_VAL_ONE
#undef CALC_VAL_ONE
#endif
#ifdef CALC_VAL_ZERO
#undef CALC_VAL_ZERO
#endif
#ifdef TO_CALC_TYPE
#undef TO_CALC_TYPE
#endif
#ifdef TO_CALC_TYPE_SAT
#undef TO_CALC_TYPE_SAT
#endif
#ifdef AS_CALC_TYPE
#undef AS_CALC_TYPE
#endif
#ifdef CALC_MAX_FUNC
#undef CALC_MAX_FUNC
#endif
#ifdef CALC_MIN_FUNC
#undef CALC_MIN_FUNC
#endif
#ifdef CALC_ABS_FUNC
#undef CALC_ABS_FUNC
#endif
#ifdef CALC_TYPE_SIZE
#undef CALC_TYPE_SIZE
#endif
#ifdef CALC_IS_FP
#undef CALC_IS_FP
#endif
#ifdef INPUT_REORDER_TYPE
#undef INPUT_REORDER_TYPE
#endif
#ifdef INPUT_REORDER_VAL_MAX
#undef INPUT_REORDER_VAL_MAX
#endif
#ifdef INPUT_REORDER_VAL_MIN
#undef INPUT_REORDER_VAL_MIN
#endif
#ifdef INPUT_REORDER_VAL_ONE
#undef INPUT_REORDER_VAL_ONE
#endif
#ifdef INPUT_REORDER_VAL_ZERO
#undef INPUT_REORDER_VAL_ZERO
#endif
#ifdef TO_INPUT_REORDER_TYPE
#undef TO_INPUT_REORDER_TYPE
#endif
#ifdef TO_INPUT_REORDER_TYPE_SAT
#undef TO_INPUT_REORDER_TYPE_SAT
#endif
#ifdef AS_INPUT_REORDER_TYPE
#undef AS_INPUT_REORDER_TYPE
#endif
#ifdef INPUT_REORDER_MAX_FUNC
#undef INPUT_REORDER_MAX_FUNC
#endif
#ifdef INPUT_REORDER_MIN_FUNC
#undef INPUT_REORDER_MIN_FUNC
#endif
#ifdef INPUT_REORDER_ABS_FUNC
#undef INPUT_REORDER_ABS_FUNC
#endif
#ifdef INPUT_REORDER_TYPE_SIZE
#undef INPUT_REORDER_TYPE_SIZE
#endif
#ifdef INPUT_REORDER_IS_FP
#undef INPUT_REORDER_IS_FP
#endif
#ifdef OUTPUT_REORDER_TYPE
#undef OUTPUT_REORDER_TYPE
#endif
#ifdef OUTPUT_REORDER_VAL_MAX
#undef OUTPUT_REORDER_VAL_MAX
#endif
#ifdef OUTPUT_REORDER_VAL_MIN
#undef OUTPUT_REORDER_VAL_MIN
#endif
#ifdef OUTPUT_REORDER_VAL_ONE
#undef OUTPUT_REORDER_VAL_ONE
#endif
#ifdef OUTPUT_REORDER_VAL_ZERO
#undef OUTPUT_REORDER_VAL_ZERO
#endif
#ifdef TO_OUTPUT_REORDER_TYPE
#undef TO_OUTPUT_REORDER_TYPE
#endif
#ifdef TO_OUTPUT_REORDER_TYPE_SAT
#undef TO_OUTPUT_REORDER_TYPE_SAT
#endif
#ifdef AS_OUTPUT_REORDER_TYPE
#undef AS_OUTPUT_REORDER_TYPE
#endif
#ifdef OUTPUT_REORDER_MAX_FUNC
#undef OUTPUT_REORDER_MAX_FUNC
#endif
#ifdef OUTPUT_REORDER_MIN_FUNC
#undef OUTPUT_REORDER_MIN_FUNC
#endif
#ifdef OUTPUT_REORDER_ABS_FUNC
#undef OUTPUT_REORDER_ABS_FUNC
#endif
#ifdef OUTPUT_REORDER_TYPE_SIZE
#undef OUTPUT_REORDER_TYPE_SIZE
#endif
#ifdef OUTPUT_REORDER_IS_FP
#undef OUTPUT_REORDER_IS_FP
#endif
#ifdef MEAN_OP
#undef MEAN_OP
#endif
#ifdef NL_M_TYPED
#undef NL_M_TYPED
#endif
#ifdef NL_N_TYPED
#undef NL_N_TYPED
#endif
#ifdef ACTIVATION_FUNC_TYPED_TYPE
#undef ACTIVATION_FUNC_TYPED_TYPE
#endif
#ifdef ACTIVATION_FUNC_TYPED_VAL_MAX
#undef ACTIVATION_FUNC_TYPED_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_TYPED_VAL_MIN
#undef ACTIVATION_FUNC_TYPED_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_TYPED_VAL_ONE
#undef ACTIVATION_FUNC_TYPED_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_TYPED_VAL_ZERO
#undef ACTIVATION_FUNC_TYPED_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_TYPED_TYPE
#undef TO_ACTIVATION_FUNC_TYPED_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_TYPED_TYPE_SAT
#undef TO_ACTIVATION_FUNC_TYPED_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_TYPED_TYPE
#undef AS_ACTIVATION_FUNC_TYPED_TYPE
#endif
#ifdef ACTIVATION_FUNC_TYPED_MAX_FUNC
#undef ACTIVATION_FUNC_TYPED_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_TYPED_MIN_FUNC
#undef ACTIVATION_FUNC_TYPED_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_TYPED_ABS_FUNC
#undef ACTIVATION_FUNC_TYPED_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_TYPED_TYPE_SIZE
#undef ACTIVATION_FUNC_TYPED_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_TYPED_IS_FP
#undef ACTIVATION_FUNC_TYPED_IS_FP
#endif
#ifdef ACTIVATION_PARAMS_TYPED
#undef ACTIVATION_PARAMS_TYPED
#endif
#ifdef ACTIVATION_FUNC_TYPED
#undef ACTIVATION_FUNC_TYPED
#endif
#ifdef ACTIVATION_TYPED
#undef ACTIVATION_TYPED
#endif
#ifdef SUB_GROUP_SIZE
#undef SUB_GROUP_SIZE
#endif
#ifdef CONVERT_TRUNCATE
#undef CONVERT_TRUNCATE
#endif
#ifdef GWS_BATCH
#undef GWS_BATCH
#endif
#ifdef GWS_FEATURE
#undef GWS_FEATURE
#endif
#ifdef GWS_YX
#undef GWS_YX
#endif

//====================================================
// Kernel template: generic_eltwise_ref 
// Kernel name: generic_eltwise_ref_9987575554347941427_0_0
#define KERNEL(name) __kernel void generic_eltwise_ref_9987575554347941427_0_0
#define KERNEL_ID generic_eltwise_ref_9987575554347941427_0_0
#define FUNC(name)  _##name##_generic_eltwise_ref_9987575554347941427_0_0
#define FUNC_CALL(name)  _##name##_generic_eltwise_ref_9987575554347941427_0_0
#define CONST_ARRAY_DECL(name) __constant size_t  _##name##_generic_eltwise_ref_9987575554347941427_0_0 []
#define CONST_ARRAY_REF(name)  _##name##_generic_eltwise_ref_9987575554347941427_0_0
#define FP64_SUPPORTED 0
#define FP16_SUPPORTED 1
#define FP16_UNIT_USED 1
#define INT8_UNIT_USED 0
#define INT32_UNIT_USED 0
#define INT64_UNIT_USED 0
#define UINT8_UNIT_USED 0
#define UINT32_UNIT_USED 0
#define UNIT_TYPE half
#define UNIT_VAL_MAX HALF_MAX
#define UNIT_VAL_MIN -UNIT_VAL_MAX
#define UNIT_VAL_ONE 1.0h
#define UNIT_VAL_ZERO 0.0h
#define TO_UNIT_TYPE(v) convert_half(v)
#define TO_UNIT_TYPE_SAT(v) convert_half(v)
#define AS_UNIT_TYPE(v) as_half(v)
#define UNIT_MAX_FUNC fmax
#define UNIT_MIN_FUNC fmin
#define UNIT_ABS_FUNC fabs
#define UNIT_TYPE_SIZE 2
#define UNIT_IS_FP 1
#define NL_M as_float(0x0)/*0.000000e+00*/
#define NL_N as_float(0x0)/*0.000000e+00*/
#define ACTIVATION_FUNC_TYPE half
#define ACTIVATION_FUNC_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_VAL_MIN -ACTIVATION_FUNC_VAL_MAX
#define ACTIVATION_FUNC_VAL_ONE 1.0h
#define ACTIVATION_FUNC_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_MAX_FUNC fmax
#define ACTIVATION_FUNC_MIN_FUNC fmin
#define ACTIVATION_FUNC_ABS_FUNC fabs
#define ACTIVATION_FUNC_TYPE_SIZE 2
#define ACTIVATION_FUNC_IS_FP 1
#define ACTIVATION_PARAMS NL_M, NL_N
#define ACTIVATION_FUNC(input, m, n) input
#define ACTIVATION(input, params) ACTIVATION_FUNC(input, params)
#define INPUT0_SIZE_X 224
#define INPUT0_SIZE_Y 224
#define INPUT0_SIZE_Z 1
#define INPUT0_SIZE_W 1
#define INPUT0_SIZE_U 1
#define INPUT0_SIZE_V 1
#define INPUT0_FEATURE_NUM 3
#define INPUT0_BATCH_NUM 1
#define INPUT0_PAD_BEFORE_SIZE_X 0
#define INPUT0_PAD_BEFORE_SIZE_Y 0
#define INPUT0_PAD_BEFORE_SIZE_Z 0
#define INPUT0_PAD_BEFORE_SIZE_W 0
#define INPUT0_PAD_BEFORE_SIZE_U 0
#define INPUT0_PAD_BEFORE_SIZE_V 0
#define INPUT0_PAD_BEFORE_FEATURE_NUM 0
#define INPUT0_PAD_BEFORE_BATCH_NUM 0
#define INPUT0_PAD_AFTER_SIZE_X 0
#define INPUT0_PAD_AFTER_SIZE_Y 0
#define INPUT0_PAD_AFTER_SIZE_Z 0
#define INPUT0_PAD_AFTER_SIZE_W 0
#define INPUT0_PAD_AFTER_SIZE_U 0
#define INPUT0_PAD_AFTER_SIZE_V 0
#define INPUT0_PAD_AFTER_FEATURE_NUM 0
#define INPUT0_PAD_AFTER_BATCH_NUM 0
#define INPUT0_LENGTH 150528
#define INPUT0_X_PITCH 1
#define INPUT0_Y_PITCH 224
#define INPUT0_Z_PITCH 1
#define INPUT0_W_PITCH 1
#define INPUT0_U_PITCH 1
#define INPUT0_V_PITCH 1
#define INPUT0_FEATURE_PITCH 50176
#define INPUT0_BATCH_PITCH 150528
#define INPUT0_GET_INDEX_SAFE(b, f, y, x) GET_DATA_INDEX_SAFE(INPUT0, b, f, y, x)
#define INPUT0_GET_INDEX(b, f, y, x) GET_DATA_INDEX(INPUT0, b, f, y, x)
#define INPUT0_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(INPUT0, b, f, y, x)
#define INPUT0_VIEW_OFFSET 0
#define INPUT0_DIMS 4
#define INPUT0_SIMPLE 1
#define INPUT0_GROUPED 0
#define INPUT0_LAYOUT_BFYX 1
#define INPUT0_TYPE half
#define INPUT0_VAL_MAX HALF_MAX
#define INPUT0_VAL_MIN -INPUT0_VAL_MAX
#define INPUT0_VAL_ONE 1.0h
#define INPUT0_VAL_ZERO 0.0h
#define TO_INPUT0_TYPE(v) convert_half(v)
#define TO_INPUT0_TYPE_SAT(v) convert_half(v)
#define AS_INPUT0_TYPE(v) as_half(v)
#define INPUT0_MAX_FUNC fmax
#define INPUT0_MIN_FUNC fmin
#define INPUT0_ABS_FUNC fabs
#define INPUT0_TYPE_SIZE 2
#define INPUT0_IS_FP 1
#define INPUT0_OFFSET 0
#define INPUT0_SIZES_DATA { 224,224,3,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(INPUT0_SIZES) = INPUT0_SIZES_DATA;
#define INPUT0_SIZES CONST_ARRAY_REF(INPUT0_SIZES)
#define INPUT0_PITCHES (size_t []){ 1,224,50176,150528,1,1,1,1,1, } 
#define INPUT0_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define INPUT0_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define INPUT1_SIZE_X 1
#define INPUT1_SIZE_Y 1
#define INPUT1_SIZE_Z 1
#define INPUT1_SIZE_W 1
#define INPUT1_SIZE_U 1
#define INPUT1_SIZE_V 1
#define INPUT1_FEATURE_NUM 3
#define INPUT1_BATCH_NUM 1
#define INPUT1_PAD_BEFORE_SIZE_X 0
#define INPUT1_PAD_BEFORE_SIZE_Y 0
#define INPUT1_PAD_BEFORE_SIZE_Z 0
#define INPUT1_PAD_BEFORE_SIZE_W 0
#define INPUT1_PAD_BEFORE_SIZE_U 0
#define INPUT1_PAD_BEFORE_SIZE_V 0
#define INPUT1_PAD_BEFORE_FEATURE_NUM 0
#define INPUT1_PAD_BEFORE_BATCH_NUM 0
#define INPUT1_PAD_AFTER_SIZE_X 0
#define INPUT1_PAD_AFTER_SIZE_Y 0
#define INPUT1_PAD_AFTER_SIZE_Z 0
#define INPUT1_PAD_AFTER_SIZE_W 0
#define INPUT1_PAD_AFTER_SIZE_U 0
#define INPUT1_PAD_AFTER_SIZE_V 0
#define INPUT1_PAD_AFTER_FEATURE_NUM 0
#define INPUT1_PAD_AFTER_BATCH_NUM 0
#define INPUT1_LENGTH 3
#define INPUT1_X_PITCH 1
#define INPUT1_Y_PITCH 1
#define INPUT1_Z_PITCH 1
#define INPUT1_W_PITCH 1
#define INPUT1_U_PITCH 1
#define INPUT1_V_PITCH 1
#define INPUT1_FEATURE_PITCH 1
#define INPUT1_BATCH_PITCH 3
#define INPUT1_GET_INDEX_SAFE(b, f, y, x) ((0 + (f)) % 3)
#define INPUT1_GET_INDEX(b, f, y, x) (0 + (f))
#define INPUT1_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(INPUT1, b, f, y, x)
#define INPUT1_VIEW_OFFSET 0
#define INPUT1_DIMS 4
#define INPUT1_SIMPLE 1
#define INPUT1_GROUPED 0
#define INPUT1_LAYOUT_BFYX 1
#define INPUT1_TYPE half
#define INPUT1_VAL_MAX HALF_MAX
#define INPUT1_VAL_MIN -INPUT1_VAL_MAX
#define INPUT1_VAL_ONE 1.0h
#define INPUT1_VAL_ZERO 0.0h
#define TO_INPUT1_TYPE(v) convert_half(v)
#define TO_INPUT1_TYPE_SAT(v) convert_half(v)
#define AS_INPUT1_TYPE(v) as_half(v)
#define INPUT1_MAX_FUNC fmax
#define INPUT1_MIN_FUNC fmin
#define INPUT1_ABS_FUNC fabs
#define INPUT1_TYPE_SIZE 2
#define INPUT1_IS_FP 1
#define INPUT1_OFFSET 0
#define INPUT1_SIZES_DATA { 1,1,3,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(INPUT1_SIZES) = INPUT1_SIZES_DATA;
#define INPUT1_SIZES CONST_ARRAY_REF(INPUT1_SIZES)
#define INPUT1_PITCHES (size_t []){ 1,1,1,3,1,1,1,1,1, } 
#define INPUT1_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define INPUT1_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OUTPUT_SIZE_X 224
#define OUTPUT_SIZE_Y 224
#define OUTPUT_SIZE_Z 1
#define OUTPUT_SIZE_W 1
#define OUTPUT_SIZE_U 1
#define OUTPUT_SIZE_V 1
#define OUTPUT_FEATURE_NUM 3
#define OUTPUT_BATCH_NUM 1
#define OUTPUT_PAD_BEFORE_SIZE_X 0
#define OUTPUT_PAD_BEFORE_SIZE_Y 0
#define OUTPUT_PAD_BEFORE_SIZE_Z 0
#define OUTPUT_PAD_BEFORE_SIZE_W 0
#define OUTPUT_PAD_BEFORE_SIZE_U 0
#define OUTPUT_PAD_BEFORE_SIZE_V 0
#define OUTPUT_PAD_BEFORE_FEATURE_NUM 0
#define OUTPUT_PAD_BEFORE_BATCH_NUM 0
#define OUTPUT_PAD_AFTER_SIZE_X 0
#define OUTPUT_PAD_AFTER_SIZE_Y 0
#define OUTPUT_PAD_AFTER_SIZE_Z 0
#define OUTPUT_PAD_AFTER_SIZE_W 0
#define OUTPUT_PAD_AFTER_SIZE_U 0
#define OUTPUT_PAD_AFTER_SIZE_V 0
#define OUTPUT_PAD_AFTER_FEATURE_NUM 0
#define OUTPUT_PAD_AFTER_BATCH_NUM 0
#define OUTPUT_LENGTH 150528
#define OUTPUT_X_PITCH 1
#define OUTPUT_Y_PITCH 224
#define OUTPUT_Z_PITCH 1
#define OUTPUT_W_PITCH 1
#define OUTPUT_U_PITCH 1
#define OUTPUT_V_PITCH 1
#define OUTPUT_FEATURE_PITCH 50176
#define OUTPUT_BATCH_PITCH 150528
#define OUTPUT_GET_INDEX_SAFE(b, f, y, x) GET_DATA_INDEX_SAFE(OUTPUT, b, f, y, x)
#define OUTPUT_GET_INDEX(b, f, y, x) GET_DATA_INDEX(OUTPUT, b, f, y, x)
#define OUTPUT_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(OUTPUT, b, f, y, x)
#define OUTPUT_VIEW_OFFSET 0
#define OUTPUT_DIMS 4
#define OUTPUT_SIMPLE 1
#define OUTPUT_GROUPED 0
#define OUTPUT_LAYOUT_BFYX 1
#define OUTPUT_TYPE half
#define OUTPUT_VAL_MAX HALF_MAX
#define OUTPUT_VAL_MIN -OUTPUT_VAL_MAX
#define OUTPUT_VAL_ONE 1.0h
#define OUTPUT_VAL_ZERO 0.0h
#define TO_OUTPUT_TYPE(v) convert_half(v)
#define TO_OUTPUT_TYPE_SAT(v) convert_half(v)
#define AS_OUTPUT_TYPE(v) as_half(v)
#define OUTPUT_MAX_FUNC fmax
#define OUTPUT_MIN_FUNC fmin
#define OUTPUT_ABS_FUNC fabs
#define OUTPUT_TYPE_SIZE 2
#define OUTPUT_IS_FP 1
#define OUTPUT_OFFSET 0
#define OUTPUT_SIZES_DATA { 224,224,3,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(OUTPUT_SIZES) = OUTPUT_SIZES_DATA;
#define OUTPUT_SIZES CONST_ARRAY_REF(OUTPUT_SIZES)
#define OUTPUT_PITCHES (size_t []){ 1,224,50176,150528,1,1,1,1,1, } 
#define OUTPUT_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OUTPUT_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OPTIONAL_SHAPE_INFO_ARG 
#define OPTIONAL_SHAPE_INFO_TENSOR 
#define ELTWISE_LAYOUT_BASED 0
#define QUANTIZATION_TERM 0
#define ELTWISE_BROADCAST 1
#define ACCUMULATOR_TYPE half
#define ACCUMULATOR_VAL_MAX HALF_MAX
#define ACCUMULATOR_VAL_MIN -ACCUMULATOR_VAL_MAX
#define ACCUMULATOR_VAL_ONE 1.0h
#define ACCUMULATOR_VAL_ZERO 0.0h
#define TO_ACCUMULATOR_TYPE(v) convert_half(v)
#define TO_ACCUMULATOR_TYPE_SAT(v) convert_half(v)
#define AS_ACCUMULATOR_TYPE(v) as_half(v)
#define ACCUMULATOR_MAX_FUNC fmax
#define ACCUMULATOR_MIN_FUNC fmin
#define ACCUMULATOR_ABS_FUNC fabs
#define ACCUMULATOR_TYPE_SIZE 2
#define ACCUMULATOR_IS_FP 1
#define ELTWISE_NO_PITCH_SAME_DIMS 0
#define INPUTS_DECLS const __global half* input0, const __global half* input1, 
#define OUTPUT_IDX_ORDER d4,d3,d2,d1
#define INPUT0_IDX_ORDER d4,d3,d2,d1
#define INPUT1_IDX_ORDER d4,d3,d2,d1
#define INPUT_0_0 input0[GET_INDEX(INPUT, 0,INPUT0_IDX_ORDER) ]
#define INPUT_0_1 input1[GET_INDEX(INPUT, 1,INPUT1_IDX_ORDER) ]
#define OPERATION0 const ACCUMULATOR_TYPE tmp0 = (ACCUMULATOR_TYPE)INPUT_0_0 * (ACCUMULATOR_TYPE)INPUT_0_1
#define DO_ELTWISE \
	OPERATION0;\
	res = tmp0;
#define GWS_BATCH 2
#define GWS_FEATURE 1
#define GWS_YX 0
#define NL_M_TYPED as_float(0x0)/*0.000000e+00*/
#define NL_N_TYPED as_float(0x0)/*0.000000e+00*/
#define ACTIVATION_FUNC_TYPED_TYPE half
#define ACTIVATION_FUNC_TYPED_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_TYPED_VAL_MIN -ACTIVATION_FUNC_TYPED_VAL_MAX
#define ACTIVATION_FUNC_TYPED_VAL_ONE 1.0h
#define ACTIVATION_FUNC_TYPED_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_TYPED_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_TYPED_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_TYPED_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_TYPED_MAX_FUNC fmax
#define ACTIVATION_FUNC_TYPED_MIN_FUNC fmin
#define ACTIVATION_FUNC_TYPED_ABS_FUNC fabs
#define ACTIVATION_FUNC_TYPED_TYPE_SIZE 2
#define ACTIVATION_FUNC_TYPED_IS_FP 1
#define ACTIVATION_PARAMS_TYPED NL_M_TYPED, NL_N_TYPED
#define ACTIVATION_FUNC_TYPED(input, m, n) input
#define ACTIVATION_TYPED(input, params) ACTIVATION_FUNC_TYPED(input, params)
#define FUSED_OP0_LOAD \
	FUSED_OP_0_INPUT0_TYPE eltwise0_data0 = eltwise0_input0[FUSED_OP_0_INPUT0_GET_INDEX_SAFE(0,d3,0,0)];
#define FUSED_OP0_ACTION \
	half res_out_0_tmp = res+eltwise0_data0;\
	half res_out_0 = convert_half(res_out_0_tmp);
#define FUSED_OPS \
	FUSED_OP0_LOAD\
	FUSED_OP0_ACTION
#define FUSED_OPS_PRELOAD \
	FUSED_OP0_LOAD
#define FUSED_OPS_CALC \
	FUSED_OP0_ACTION
#define FUSED_OPS_RESULT res_out_0
#define FUSED_OPS_CAN_USE_PRELOAD 1
#define FUSED_OP_0_INPUT0_SIZE_X 1
#define FUSED_OP_0_INPUT0_SIZE_Y 1
#define FUSED_OP_0_INPUT0_SIZE_Z 1
#define FUSED_OP_0_INPUT0_SIZE_W 1
#define FUSED_OP_0_INPUT0_SIZE_U 1
#define FUSED_OP_0_INPUT0_SIZE_V 1
#define FUSED_OP_0_INPUT0_FEATURE_NUM 3
#define FUSED_OP_0_INPUT0_BATCH_NUM 1
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_X 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Y 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Z 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_W 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_U 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_V 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_FEATURE_NUM 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_BATCH_NUM 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_X 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Y 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Z 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_W 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_U 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_V 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_FEATURE_NUM 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_BATCH_NUM 0
#define FUSED_OP_0_INPUT0_LENGTH 3
#define FUSED_OP_0_INPUT0_X_PITCH 1
#define FUSED_OP_0_INPUT0_Y_PITCH 1
#define FUSED_OP_0_INPUT0_Z_PITCH 1
#define FUSED_OP_0_INPUT0_W_PITCH 1
#define FUSED_OP_0_INPUT0_U_PITCH 1
#define FUSED_OP_0_INPUT0_V_PITCH 1
#define FUSED_OP_0_INPUT0_FEATURE_PITCH 1
#define FUSED_OP_0_INPUT0_BATCH_PITCH 3
#define FUSED_OP_0_INPUT0_GET_INDEX_SAFE(b, f, y, x) ((0 + (f)) % 3)
#define FUSED_OP_0_INPUT0_GET_INDEX(b, f, y, x) (0 + (f))
#define FUSED_OP_0_INPUT0_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(FUSED_OP_0_INPUT0, b, f, y, x)
#define FUSED_OP_0_INPUT0_VIEW_OFFSET 0
#define FUSED_OP_0_INPUT0_DIMS 4
#define FUSED_OP_0_INPUT0_SIMPLE 1
#define FUSED_OP_0_INPUT0_GROUPED 0
#define FUSED_OP_0_INPUT0_LAYOUT_BFYX 1
#define FUSED_OP_0_INPUT0_TYPE half
#define FUSED_OP_0_INPUT0_VAL_MAX HALF_MAX
#define FUSED_OP_0_INPUT0_VAL_MIN -FUSED_OP_0_INPUT0_VAL_MAX
#define FUSED_OP_0_INPUT0_VAL_ONE 1.0h
#define FUSED_OP_0_INPUT0_VAL_ZERO 0.0h
#define TO_FUSED_OP_0_INPUT0_TYPE(v) convert_half(v)
#define TO_FUSED_OP_0_INPUT0_TYPE_SAT(v) convert_half(v)
#define AS_FUSED_OP_0_INPUT0_TYPE(v) as_half(v)
#define FUSED_OP_0_INPUT0_MAX_FUNC fmax
#define FUSED_OP_0_INPUT0_MIN_FUNC fmin
#define FUSED_OP_0_INPUT0_ABS_FUNC fabs
#define FUSED_OP_0_INPUT0_TYPE_SIZE 2
#define FUSED_OP_0_INPUT0_IS_FP 1
#define FUSED_OP_0_INPUT0_OFFSET 0
#define FUSED_OP_0_INPUT0_SIZES_DATA { 1,1,3,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(FUSED_OP_0_INPUT0_SIZES) = FUSED_OP_0_INPUT0_SIZES_DATA;
#define FUSED_OP_0_INPUT0_SIZES CONST_ARRAY_REF(FUSED_OP_0_INPUT0_SIZES)
#define FUSED_OP_0_INPUT0_PITCHES (size_t []){ 1,1,1,3,1,1,1,1,1, } 
#define FUSED_OP_0_INPUT0_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP_0_INPUT0_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP_0_OUTPUT_SIZE_X 224
#define FUSED_OP_0_OUTPUT_SIZE_Y 224
#define FUSED_OP_0_OUTPUT_SIZE_Z 1
#define FUSED_OP_0_OUTPUT_SIZE_W 1
#define FUSED_OP_0_OUTPUT_SIZE_U 1
#define FUSED_OP_0_OUTPUT_SIZE_V 1
#define FUSED_OP_0_OUTPUT_FEATURE_NUM 3
#define FUSED_OP_0_OUTPUT_BATCH_NUM 1
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_X 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Y 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Z 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_W 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_U 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_V 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_FEATURE_NUM 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_BATCH_NUM 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_X 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Y 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Z 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_W 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_U 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_V 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_FEATURE_NUM 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_BATCH_NUM 0
#define FUSED_OP_0_OUTPUT_LENGTH 150528
#define FUSED_OP_0_OUTPUT_X_PITCH 1
#define FUSED_OP_0_OUTPUT_Y_PITCH 224
#define FUSED_OP_0_OUTPUT_Z_PITCH 1
#define FUSED_OP_0_OUTPUT_W_PITCH 1
#define FUSED_OP_0_OUTPUT_U_PITCH 1
#define FUSED_OP_0_OUTPUT_V_PITCH 1
#define FUSED_OP_0_OUTPUT_FEATURE_PITCH 50176
#define FUSED_OP_0_OUTPUT_BATCH_PITCH 150528
#define FUSED_OP_0_OUTPUT_GET_INDEX_SAFE(b, f, y, x) GET_DATA_INDEX_SAFE(FUSED_OP_0_OUTPUT, b, f, y, x)
#define FUSED_OP_0_OUTPUT_GET_INDEX(b, f, y, x) GET_DATA_INDEX(FUSED_OP_0_OUTPUT, b, f, y, x)
#define FUSED_OP_0_OUTPUT_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(FUSED_OP_0_OUTPUT, b, f, y, x)
#define FUSED_OP_0_OUTPUT_VIEW_OFFSET 0
#define FUSED_OP_0_OUTPUT_DIMS 4
#define FUSED_OP_0_OUTPUT_SIMPLE 1
#define FUSED_OP_0_OUTPUT_GROUPED 0
#define FUSED_OP_0_OUTPUT_LAYOUT_BFYX 1
#define FUSED_OP_0_OUTPUT_TYPE half
#define FUSED_OP_0_OUTPUT_VAL_MAX HALF_MAX
#define FUSED_OP_0_OUTPUT_VAL_MIN -FUSED_OP_0_OUTPUT_VAL_MAX
#define FUSED_OP_0_OUTPUT_VAL_ONE 1.0h
#define FUSED_OP_0_OUTPUT_VAL_ZERO 0.0h
#define TO_FUSED_OP_0_OUTPUT_TYPE(v) convert_half(v)
#define TO_FUSED_OP_0_OUTPUT_TYPE_SAT(v) convert_half(v)
#define AS_FUSED_OP_0_OUTPUT_TYPE(v) as_half(v)
#define FUSED_OP_0_OUTPUT_MAX_FUNC fmax
#define FUSED_OP_0_OUTPUT_MIN_FUNC fmin
#define FUSED_OP_0_OUTPUT_ABS_FUNC fabs
#define FUSED_OP_0_OUTPUT_TYPE_SIZE 2
#define FUSED_OP_0_OUTPUT_IS_FP 1
#define FUSED_OP_0_OUTPUT_OFFSET 0
#define FUSED_OP_0_OUTPUT_SIZES_DATA { 224,224,3,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(FUSED_OP_0_OUTPUT_SIZES) = FUSED_OP_0_OUTPUT_SIZES_DATA;
#define FUSED_OP_0_OUTPUT_SIZES CONST_ARRAY_REF(FUSED_OP_0_OUTPUT_SIZES)
#define FUSED_OP_0_OUTPUT_PITCHES (size_t []){ 1,224,50176,150528,1,1,1,1,1, } 
#define FUSED_OP_0_OUTPUT_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP_0_OUTPUT_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP0_DECLS \
	const __global half* eltwise0_input0
#define FUSED_OP0_ARGS \
	eltwise0_input0
#define FUSED_OPS_DECLS \
	FUSED_OP0_DECLS
#define FUSED_OPS_ARGS \
	FUSED_OP0_ARGS
#define HAS_FUSED_OPS 1
#define HAS_FUSED_OPS_DECLS 1


#if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM || ELTWISE_BROADCAST
 #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX_SAFE)(idx_order)
#elif ELTWISE_NO_PITCH_SAME_DIMS
 #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _OFFSET) + idx_order
#else
 #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX)(idx_order)
#endif
KERNEL(eltwise)(
 OPTIONAL_SHAPE_INFO_ARG
 INPUTS_DECLS
 __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
 , FUSED_OPS_DECLS
#endif
#if IS_DYNAMIC_CROP
 , int runtime_offset
#endif
)
{
#if OUTPUT_DIMS == 8
 #if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM || ELTWISE_BROADCAST
 uint data_idx = get_global_id(GWS_YX);
 const uint d1 = data_idx % OUTPUT_SIZE_X;
 data_idx = data_idx / OUTPUT_SIZE_X;
 const uint d2 = data_idx % OUTPUT_SIZE_Y;
 data_idx = data_idx / OUTPUT_SIZE_Y;
 const uint d3 = data_idx % OUTPUT_SIZE_Z;
 data_idx = data_idx / OUTPUT_SIZE_Z;
 const uint d4 = data_idx % OUTPUT_SIZE_W;
 data_idx = data_idx / OUTPUT_SIZE_W;
 const uint d5 = data_idx % OUTPUT_SIZE_U;
 data_idx = data_idx / OUTPUT_SIZE_U;
 const uint d6 = data_idx % OUTPUT_SIZE_V;
 const uint d7 = get_global_id(GWS_FEATURE);
 const uint d8 = get_global_id(GWS_BATCH);
 uint output_offset = OUTPUT_GET_INDEX(d8, d7, d6, d5, d4, d3, d2, d1);
 #elif ELTWISE_NO_PITCH_SAME_DIMS
 const uint d1 = get_global_id(0);
 uint output_offset = OUTPUT_OFFSET + d1;
 #else
 const uint d1 = get_global_id(0);
 const uint d2 = (uint)get_global_id(1) % OUTPUT_SIZES[1];
 const uint d3 = (uint)get_global_id(1) / OUTPUT_SIZES[1] % OUTPUT_SIZES[2];
 const uint d4 = (uint)get_global_id(1) / OUTPUT_SIZES[1] / OUTPUT_SIZES[2] % OUTPUT_SIZES[3];
 const uint d5 = (uint)get_global_id(1) / OUTPUT_SIZES[1] / OUTPUT_SIZES[2] / OUTPUT_SIZES[3] % OUTPUT_SIZES[4];
 const uint d6 = (uint)get_global_id(1) / OUTPUT_SIZES[1] / OUTPUT_SIZES[2] / OUTPUT_SIZES[3] / OUTPUT_SIZES[4];
 const uint d7 = (uint)get_global_id(2) % OUTPUT_SIZES[6];
 const uint d8 = (uint)get_global_id(2) / OUTPUT_SIZES[6];
 uint output_offset = OUTPUT_GET_INDEX(d8, d7, d6, d5, d4, d3, d2, d1);
 #endif
#elif OUTPUT_DIMS == 7
 #if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM || ELTWISE_BROADCAST
 uint data_idx = get_global_id(GWS_YX);
 const uint d1 = data_idx % OUTPUT_SIZE_X;
 data_idx = data_idx / OUTPUT_SIZE_X;
 const uint d2 = data_idx % OUTPUT_SIZE_Y;
 data_idx = data_idx / OUTPUT_SIZE_Y;
 const uint d3 = data_idx % OUTPUT_SIZE_Z;
 data_idx = data_idx / OUTPUT_SIZE_Z;
 const uint d4 = data_idx % OUTPUT_SIZE_W;
 data_idx = data_idx / OUTPUT_SIZE_W;
 const uint d5 = data_idx % OUTPUT_SIZE_U;
 const uint d6 = get_global_id(GWS_FEATURE);
 const uint d7 = get_global_id(GWS_BATCH);
 uint output_offset = OUTPUT_GET_INDEX(d7, d6, d5, d4, d3, d2, d1);
 #elif ELTWISE_NO_PITCH_SAME_DIMS
 const uint d1 = get_global_id(0);
 uint output_offset = OUTPUT_OFFSET + d1;
 #else
 const uint d1 = get_global_id(0);
 const uint d2 = (uint)get_global_id(1) % OUTPUT_SIZES[1];
 const uint d3 = (uint)get_global_id(1) / OUTPUT_SIZES[1] % OUTPUT_SIZES[2];
 const uint d4 = (uint)get_global_id(1) / OUTPUT_SIZES[1] / OUTPUT_SIZES[2] % OUTPUT_SIZES[3];
 const uint d5 = (uint)get_global_id(1) / OUTPUT_SIZES[1] / OUTPUT_SIZES[2] / OUTPUT_SIZES[3];
 const uint d6 = (uint)get_global_id(2) % OUTPUT_SIZES[5];
 const uint d7 = (uint)get_global_id(2) / OUTPUT_SIZES[5];
 uint output_offset = OUTPUT_GET_INDEX(d7, d6, d5, d4, d3, d2, d1);
 #endif
#elif OUTPUT_DIMS == 6
 #if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM || ELTWISE_BROADCAST
 uint data_idx = get_global_id(GWS_YX);
 const uint d1 = data_idx % OUTPUT_SIZE_X;
 data_idx = data_idx / OUTPUT_SIZE_X;
 const uint d2 = data_idx % OUTPUT_SIZE_Y;
 data_idx = data_idx / OUTPUT_SIZE_Y;
 const uint d3 = data_idx % OUTPUT_SIZE_Z;
 data_idx = data_idx / OUTPUT_SIZE_Z;
 const uint d4 = data_idx % OUTPUT_SIZE_W;
 const uint d5 = get_global_id(GWS_FEATURE);
 const uint d6 = get_global_id(GWS_BATCH);
 uint output_offset = OUTPUT_GET_INDEX(d6, d5, d4, d3, d2, d1);
 #elif ELTWISE_NO_PITCH_SAME_DIMS
 const uint d1 = get_global_id(0);
 uint output_offset = OUTPUT_OFFSET + d1;
 #else
 const uint d1 = get_global_id(0);
 const uint d2 = (uint)get_global_id(1) % OUTPUT_SIZES[1];
 const uint d3 = (uint)get_global_id(1) / OUTPUT_SIZES[1] % OUTPUT_SIZES[2];
 const uint d4 = (uint)get_global_id(1) / OUTPUT_SIZES[1] / OUTPUT_SIZES[2];
 const uint d5 = (uint)get_global_id(2) % OUTPUT_SIZES[4];
 const uint d6 = (uint)get_global_id(2) / OUTPUT_SIZES[4];
 uint output_offset = OUTPUT_GET_INDEX(d6, d5, d4, d3, d2, d1);
 #endif
#elif OUTPUT_DIMS == 5
 #if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM || ELTWISE_BROADCAST
 uint data_idx = get_global_id(GWS_YX);
 const uint d1 = data_idx % OUTPUT_SIZE_X;
 data_idx = data_idx / OUTPUT_SIZE_X;
 const uint d2 = data_idx % OUTPUT_SIZE_Y;
 data_idx = data_idx / OUTPUT_SIZE_Y;
 const uint d3 = data_idx % OUTPUT_SIZE_Z;
 const uint d4 = get_global_id(GWS_FEATURE);
 const uint d5 = get_global_id(GWS_BATCH);
 uint output_offset = OUTPUT_GET_INDEX(d5, d4, d3, d2, d1);
 #elif ELTWISE_NO_PITCH_SAME_DIMS
 const uint d1 = get_global_id(0);
 uint output_offset = OUTPUT_OFFSET + d1;
 #else
 const uint d1 = get_global_id(0);
 const uint d2 = (uint)get_global_id(1) % OUTPUT_SIZES[1];
 const uint d3 = (uint)get_global_id(1) / OUTPUT_SIZES[1];
 const uint d4 = (uint)get_global_id(2) % OUTPUT_SIZES[3];
 const uint d5 = (uint)get_global_id(2) / OUTPUT_SIZES[3];
 uint output_offset = OUTPUT_GET_INDEX(d5, d4, d3, d2, d1);
 #endif
#else
 #if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM || ELTWISE_BROADCAST
 const uint d1 = (uint)get_global_id(GWS_YX) % OUTPUT_SIZE_X;
 const uint d2 = (uint)get_global_id(GWS_YX) / OUTPUT_SIZE_X;
 const uint d3 = (uint)get_global_id(GWS_FEATURE);
 const uint d4 = (uint)get_global_id(GWS_BATCH);
 uint output_offset = GET_INDEX(OUTPUT,, OUTPUT_IDX_ORDER);
 #elif ELTWISE_NO_PITCH_SAME_DIMS
 const uint d1 = get_global_id(0);
 uint output_offset = OUTPUT_OFFSET + d1;
 #else
 const uint d1 = get_global_id(0);
 const uint d2 = get_global_id(1);
 const uint d3 = (uint)get_global_id(2) % OUTPUT_SIZES[2];
 const uint d4 = (uint)get_global_id(2) / OUTPUT_SIZES[2];
 uint output_offset = GET_INDEX(OUTPUT,, OUTPUT_IDX_ORDER);
 #endif
#endif
 ACCUMULATOR_TYPE res;
 DO_ELTWISE;
#if HAS_FUSED_OPS
 FUSED_OPS;
 OUTPUT_TYPE out = FUSED_OPS_RESULT;
#else
 #define out res
#endif
#if QUANTIZATION_TERM && !OUTPUT_IS_FP
 output[output_offset] = TO_OUTPUT_TYPE_SAT(ACTIVATION(out, ACTIVATION_PARAMS));
#else
 output[output_offset] = TO_OUTPUT_TYPE(ACTIVATION_TYPED(out, ACTIVATION_PARAMS_TYPED));
#endif
}
#ifdef GET_INDEX
#undef GET_INDEX
#endif
#ifdef GET_INDEX
#undef GET_INDEX
#endif
#ifdef GET_INDEX
#undef GET_INDEX
#endif
#ifdef out
#undef out
#endif
#undef KERNEL
#undef KERNEL_ID
#undef FUNC
#undef FUNC_CALL
#undef CONST_ARRAY_DECL
#undef CONST_ARRAY_REF
#ifdef FP64_SUPPORTED
#undef FP64_SUPPORTED
#endif
#ifdef FP16_SUPPORTED
#undef FP16_SUPPORTED
#endif
#ifdef FP16_UNIT_USED
#undef FP16_UNIT_USED
#endif
#ifdef INT8_UNIT_USED
#undef INT8_UNIT_USED
#endif
#ifdef INT32_UNIT_USED
#undef INT32_UNIT_USED
#endif
#ifdef INT64_UNIT_USED
#undef INT64_UNIT_USED
#endif
#ifdef UINT8_UNIT_USED
#undef UINT8_UNIT_USED
#endif
#ifdef UINT32_UNIT_USED
#undef UINT32_UNIT_USED
#endif
#ifdef UNIT_TYPE
#undef UNIT_TYPE
#endif
#ifdef UNIT_VAL_MAX
#undef UNIT_VAL_MAX
#endif
#ifdef UNIT_VAL_MIN
#undef UNIT_VAL_MIN
#endif
#ifdef UNIT_VAL_ONE
#undef UNIT_VAL_ONE
#endif
#ifdef UNIT_VAL_ZERO
#undef UNIT_VAL_ZERO
#endif
#ifdef TO_UNIT_TYPE
#undef TO_UNIT_TYPE
#endif
#ifdef TO_UNIT_TYPE_SAT
#undef TO_UNIT_TYPE_SAT
#endif
#ifdef AS_UNIT_TYPE
#undef AS_UNIT_TYPE
#endif
#ifdef UNIT_MAX_FUNC
#undef UNIT_MAX_FUNC
#endif
#ifdef UNIT_MIN_FUNC
#undef UNIT_MIN_FUNC
#endif
#ifdef UNIT_ABS_FUNC
#undef UNIT_ABS_FUNC
#endif
#ifdef UNIT_TYPE_SIZE
#undef UNIT_TYPE_SIZE
#endif
#ifdef UNIT_IS_FP
#undef UNIT_IS_FP
#endif
#ifdef NL_M
#undef NL_M
#endif
#ifdef NL_N
#undef NL_N
#endif
#ifdef ACTIVATION_FUNC_TYPE
#undef ACTIVATION_FUNC_TYPE
#endif
#ifdef ACTIVATION_FUNC_VAL_MAX
#undef ACTIVATION_FUNC_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_VAL_MIN
#undef ACTIVATION_FUNC_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_VAL_ONE
#undef ACTIVATION_FUNC_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_VAL_ZERO
#undef ACTIVATION_FUNC_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_TYPE
#undef TO_ACTIVATION_FUNC_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_TYPE_SAT
#undef TO_ACTIVATION_FUNC_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_TYPE
#undef AS_ACTIVATION_FUNC_TYPE
#endif
#ifdef ACTIVATION_FUNC_MAX_FUNC
#undef ACTIVATION_FUNC_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_MIN_FUNC
#undef ACTIVATION_FUNC_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_ABS_FUNC
#undef ACTIVATION_FUNC_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_TYPE_SIZE
#undef ACTIVATION_FUNC_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_IS_FP
#undef ACTIVATION_FUNC_IS_FP
#endif
#ifdef ACTIVATION_PARAMS
#undef ACTIVATION_PARAMS
#endif
#ifdef ACTIVATION_FUNC
#undef ACTIVATION_FUNC
#endif
#ifdef ACTIVATION
#undef ACTIVATION
#endif
#ifdef INPUT0_SIZE_X
#undef INPUT0_SIZE_X
#endif
#ifdef INPUT0_SIZE_Y
#undef INPUT0_SIZE_Y
#endif
#ifdef INPUT0_SIZE_Z
#undef INPUT0_SIZE_Z
#endif
#ifdef INPUT0_SIZE_W
#undef INPUT0_SIZE_W
#endif
#ifdef INPUT0_SIZE_U
#undef INPUT0_SIZE_U
#endif
#ifdef INPUT0_SIZE_V
#undef INPUT0_SIZE_V
#endif
#ifdef INPUT0_FEATURE_NUM
#undef INPUT0_FEATURE_NUM
#endif
#ifdef INPUT0_BATCH_NUM
#undef INPUT0_BATCH_NUM
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_X
#undef INPUT0_PAD_BEFORE_SIZE_X
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_Y
#undef INPUT0_PAD_BEFORE_SIZE_Y
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_Z
#undef INPUT0_PAD_BEFORE_SIZE_Z
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_W
#undef INPUT0_PAD_BEFORE_SIZE_W
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_U
#undef INPUT0_PAD_BEFORE_SIZE_U
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_V
#undef INPUT0_PAD_BEFORE_SIZE_V
#endif
#ifdef INPUT0_PAD_BEFORE_FEATURE_NUM
#undef INPUT0_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef INPUT0_PAD_BEFORE_BATCH_NUM
#undef INPUT0_PAD_BEFORE_BATCH_NUM
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_X
#undef INPUT0_PAD_AFTER_SIZE_X
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_Y
#undef INPUT0_PAD_AFTER_SIZE_Y
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_Z
#undef INPUT0_PAD_AFTER_SIZE_Z
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_W
#undef INPUT0_PAD_AFTER_SIZE_W
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_U
#undef INPUT0_PAD_AFTER_SIZE_U
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_V
#undef INPUT0_PAD_AFTER_SIZE_V
#endif
#ifdef INPUT0_PAD_AFTER_FEATURE_NUM
#undef INPUT0_PAD_AFTER_FEATURE_NUM
#endif
#ifdef INPUT0_PAD_AFTER_BATCH_NUM
#undef INPUT0_PAD_AFTER_BATCH_NUM
#endif
#ifdef INPUT0_LENGTH
#undef INPUT0_LENGTH
#endif
#ifdef INPUT0_X_PITCH
#undef INPUT0_X_PITCH
#endif
#ifdef INPUT0_Y_PITCH
#undef INPUT0_Y_PITCH
#endif
#ifdef INPUT0_Z_PITCH
#undef INPUT0_Z_PITCH
#endif
#ifdef INPUT0_W_PITCH
#undef INPUT0_W_PITCH
#endif
#ifdef INPUT0_U_PITCH
#undef INPUT0_U_PITCH
#endif
#ifdef INPUT0_V_PITCH
#undef INPUT0_V_PITCH
#endif
#ifdef INPUT0_FEATURE_PITCH
#undef INPUT0_FEATURE_PITCH
#endif
#ifdef INPUT0_BATCH_PITCH
#undef INPUT0_BATCH_PITCH
#endif
#ifdef INPUT0_GET_INDEX_SAFE
#undef INPUT0_GET_INDEX_SAFE
#endif
#ifdef INPUT0_GET_INDEX
#undef INPUT0_GET_INDEX
#endif
#ifdef INPUT0_GET_INDEX_RAW
#undef INPUT0_GET_INDEX_RAW
#endif
#ifdef INPUT0_VIEW_OFFSET
#undef INPUT0_VIEW_OFFSET
#endif
#ifdef INPUT0_DIMS
#undef INPUT0_DIMS
#endif
#ifdef INPUT0_SIMPLE
#undef INPUT0_SIMPLE
#endif
#ifdef INPUT0_GROUPED
#undef INPUT0_GROUPED
#endif
#ifdef INPUT0_LAYOUT_BFYX
#undef INPUT0_LAYOUT_BFYX
#endif
#ifdef INPUT0_TYPE
#undef INPUT0_TYPE
#endif
#ifdef INPUT0_VAL_MAX
#undef INPUT0_VAL_MAX
#endif
#ifdef INPUT0_VAL_MIN
#undef INPUT0_VAL_MIN
#endif
#ifdef INPUT0_VAL_ONE
#undef INPUT0_VAL_ONE
#endif
#ifdef INPUT0_VAL_ZERO
#undef INPUT0_VAL_ZERO
#endif
#ifdef TO_INPUT0_TYPE
#undef TO_INPUT0_TYPE
#endif
#ifdef TO_INPUT0_TYPE_SAT
#undef TO_INPUT0_TYPE_SAT
#endif
#ifdef AS_INPUT0_TYPE
#undef AS_INPUT0_TYPE
#endif
#ifdef INPUT0_MAX_FUNC
#undef INPUT0_MAX_FUNC
#endif
#ifdef INPUT0_MIN_FUNC
#undef INPUT0_MIN_FUNC
#endif
#ifdef INPUT0_ABS_FUNC
#undef INPUT0_ABS_FUNC
#endif
#ifdef INPUT0_TYPE_SIZE
#undef INPUT0_TYPE_SIZE
#endif
#ifdef INPUT0_IS_FP
#undef INPUT0_IS_FP
#endif
#ifdef INPUT0_OFFSET
#undef INPUT0_OFFSET
#endif
#ifdef INPUT0_SIZES_DATA
#undef INPUT0_SIZES_DATA
#endif
#ifdef INPUT0_PITCHES
#undef INPUT0_PITCHES
#endif
#ifdef INPUT0_PAD_BEFORE
#undef INPUT0_PAD_BEFORE
#endif
#ifdef INPUT0_PAD_AFTER
#undef INPUT0_PAD_AFTER
#endif
#ifdef INPUT1_SIZE_X
#undef INPUT1_SIZE_X
#endif
#ifdef INPUT1_SIZE_Y
#undef INPUT1_SIZE_Y
#endif
#ifdef INPUT1_SIZE_Z
#undef INPUT1_SIZE_Z
#endif
#ifdef INPUT1_SIZE_W
#undef INPUT1_SIZE_W
#endif
#ifdef INPUT1_SIZE_U
#undef INPUT1_SIZE_U
#endif
#ifdef INPUT1_SIZE_V
#undef INPUT1_SIZE_V
#endif
#ifdef INPUT1_FEATURE_NUM
#undef INPUT1_FEATURE_NUM
#endif
#ifdef INPUT1_BATCH_NUM
#undef INPUT1_BATCH_NUM
#endif
#ifdef INPUT1_PAD_BEFORE_SIZE_X
#undef INPUT1_PAD_BEFORE_SIZE_X
#endif
#ifdef INPUT1_PAD_BEFORE_SIZE_Y
#undef INPUT1_PAD_BEFORE_SIZE_Y
#endif
#ifdef INPUT1_PAD_BEFORE_SIZE_Z
#undef INPUT1_PAD_BEFORE_SIZE_Z
#endif
#ifdef INPUT1_PAD_BEFORE_SIZE_W
#undef INPUT1_PAD_BEFORE_SIZE_W
#endif
#ifdef INPUT1_PAD_BEFORE_SIZE_U
#undef INPUT1_PAD_BEFORE_SIZE_U
#endif
#ifdef INPUT1_PAD_BEFORE_SIZE_V
#undef INPUT1_PAD_BEFORE_SIZE_V
#endif
#ifdef INPUT1_PAD_BEFORE_FEATURE_NUM
#undef INPUT1_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef INPUT1_PAD_BEFORE_BATCH_NUM
#undef INPUT1_PAD_BEFORE_BATCH_NUM
#endif
#ifdef INPUT1_PAD_AFTER_SIZE_X
#undef INPUT1_PAD_AFTER_SIZE_X
#endif
#ifdef INPUT1_PAD_AFTER_SIZE_Y
#undef INPUT1_PAD_AFTER_SIZE_Y
#endif
#ifdef INPUT1_PAD_AFTER_SIZE_Z
#undef INPUT1_PAD_AFTER_SIZE_Z
#endif
#ifdef INPUT1_PAD_AFTER_SIZE_W
#undef INPUT1_PAD_AFTER_SIZE_W
#endif
#ifdef INPUT1_PAD_AFTER_SIZE_U
#undef INPUT1_PAD_AFTER_SIZE_U
#endif
#ifdef INPUT1_PAD_AFTER_SIZE_V
#undef INPUT1_PAD_AFTER_SIZE_V
#endif
#ifdef INPUT1_PAD_AFTER_FEATURE_NUM
#undef INPUT1_PAD_AFTER_FEATURE_NUM
#endif
#ifdef INPUT1_PAD_AFTER_BATCH_NUM
#undef INPUT1_PAD_AFTER_BATCH_NUM
#endif
#ifdef INPUT1_LENGTH
#undef INPUT1_LENGTH
#endif
#ifdef INPUT1_X_PITCH
#undef INPUT1_X_PITCH
#endif
#ifdef INPUT1_Y_PITCH
#undef INPUT1_Y_PITCH
#endif
#ifdef INPUT1_Z_PITCH
#undef INPUT1_Z_PITCH
#endif
#ifdef INPUT1_W_PITCH
#undef INPUT1_W_PITCH
#endif
#ifdef INPUT1_U_PITCH
#undef INPUT1_U_PITCH
#endif
#ifdef INPUT1_V_PITCH
#undef INPUT1_V_PITCH
#endif
#ifdef INPUT1_FEATURE_PITCH
#undef INPUT1_FEATURE_PITCH
#endif
#ifdef INPUT1_BATCH_PITCH
#undef INPUT1_BATCH_PITCH
#endif
#ifdef INPUT1_GET_INDEX_SAFE
#undef INPUT1_GET_INDEX_SAFE
#endif
#ifdef INPUT1_GET_INDEX
#undef INPUT1_GET_INDEX
#endif
#ifdef INPUT1_GET_INDEX_RAW
#undef INPUT1_GET_INDEX_RAW
#endif
#ifdef INPUT1_VIEW_OFFSET
#undef INPUT1_VIEW_OFFSET
#endif
#ifdef INPUT1_DIMS
#undef INPUT1_DIMS
#endif
#ifdef INPUT1_SIMPLE
#undef INPUT1_SIMPLE
#endif
#ifdef INPUT1_GROUPED
#undef INPUT1_GROUPED
#endif
#ifdef INPUT1_LAYOUT_BFYX
#undef INPUT1_LAYOUT_BFYX
#endif
#ifdef INPUT1_TYPE
#undef INPUT1_TYPE
#endif
#ifdef INPUT1_VAL_MAX
#undef INPUT1_VAL_MAX
#endif
#ifdef INPUT1_VAL_MIN
#undef INPUT1_VAL_MIN
#endif
#ifdef INPUT1_VAL_ONE
#undef INPUT1_VAL_ONE
#endif
#ifdef INPUT1_VAL_ZERO
#undef INPUT1_VAL_ZERO
#endif
#ifdef TO_INPUT1_TYPE
#undef TO_INPUT1_TYPE
#endif
#ifdef TO_INPUT1_TYPE_SAT
#undef TO_INPUT1_TYPE_SAT
#endif
#ifdef AS_INPUT1_TYPE
#undef AS_INPUT1_TYPE
#endif
#ifdef INPUT1_MAX_FUNC
#undef INPUT1_MAX_FUNC
#endif
#ifdef INPUT1_MIN_FUNC
#undef INPUT1_MIN_FUNC
#endif
#ifdef INPUT1_ABS_FUNC
#undef INPUT1_ABS_FUNC
#endif
#ifdef INPUT1_TYPE_SIZE
#undef INPUT1_TYPE_SIZE
#endif
#ifdef INPUT1_IS_FP
#undef INPUT1_IS_FP
#endif
#ifdef INPUT1_OFFSET
#undef INPUT1_OFFSET
#endif
#ifdef INPUT1_SIZES_DATA
#undef INPUT1_SIZES_DATA
#endif
#ifdef INPUT1_PITCHES
#undef INPUT1_PITCHES
#endif
#ifdef INPUT1_PAD_BEFORE
#undef INPUT1_PAD_BEFORE
#endif
#ifdef INPUT1_PAD_AFTER
#undef INPUT1_PAD_AFTER
#endif
#ifdef OUTPUT_SIZE_X
#undef OUTPUT_SIZE_X
#endif
#ifdef OUTPUT_SIZE_Y
#undef OUTPUT_SIZE_Y
#endif
#ifdef OUTPUT_SIZE_Z
#undef OUTPUT_SIZE_Z
#endif
#ifdef OUTPUT_SIZE_W
#undef OUTPUT_SIZE_W
#endif
#ifdef OUTPUT_SIZE_U
#undef OUTPUT_SIZE_U
#endif
#ifdef OUTPUT_SIZE_V
#undef OUTPUT_SIZE_V
#endif
#ifdef OUTPUT_FEATURE_NUM
#undef OUTPUT_FEATURE_NUM
#endif
#ifdef OUTPUT_BATCH_NUM
#undef OUTPUT_BATCH_NUM
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_X
#undef OUTPUT_PAD_BEFORE_SIZE_X
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_Y
#undef OUTPUT_PAD_BEFORE_SIZE_Y
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_Z
#undef OUTPUT_PAD_BEFORE_SIZE_Z
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_W
#undef OUTPUT_PAD_BEFORE_SIZE_W
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_U
#undef OUTPUT_PAD_BEFORE_SIZE_U
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_V
#undef OUTPUT_PAD_BEFORE_SIZE_V
#endif
#ifdef OUTPUT_PAD_BEFORE_FEATURE_NUM
#undef OUTPUT_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef OUTPUT_PAD_BEFORE_BATCH_NUM
#undef OUTPUT_PAD_BEFORE_BATCH_NUM
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_X
#undef OUTPUT_PAD_AFTER_SIZE_X
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_Y
#undef OUTPUT_PAD_AFTER_SIZE_Y
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_Z
#undef OUTPUT_PAD_AFTER_SIZE_Z
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_W
#undef OUTPUT_PAD_AFTER_SIZE_W
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_U
#undef OUTPUT_PAD_AFTER_SIZE_U
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_V
#undef OUTPUT_PAD_AFTER_SIZE_V
#endif
#ifdef OUTPUT_PAD_AFTER_FEATURE_NUM
#undef OUTPUT_PAD_AFTER_FEATURE_NUM
#endif
#ifdef OUTPUT_PAD_AFTER_BATCH_NUM
#undef OUTPUT_PAD_AFTER_BATCH_NUM
#endif
#ifdef OUTPUT_LENGTH
#undef OUTPUT_LENGTH
#endif
#ifdef OUTPUT_X_PITCH
#undef OUTPUT_X_PITCH
#endif
#ifdef OUTPUT_Y_PITCH
#undef OUTPUT_Y_PITCH
#endif
#ifdef OUTPUT_Z_PITCH
#undef OUTPUT_Z_PITCH
#endif
#ifdef OUTPUT_W_PITCH
#undef OUTPUT_W_PITCH
#endif
#ifdef OUTPUT_U_PITCH
#undef OUTPUT_U_PITCH
#endif
#ifdef OUTPUT_V_PITCH
#undef OUTPUT_V_PITCH
#endif
#ifdef OUTPUT_FEATURE_PITCH
#undef OUTPUT_FEATURE_PITCH
#endif
#ifdef OUTPUT_BATCH_PITCH
#undef OUTPUT_BATCH_PITCH
#endif
#ifdef OUTPUT_GET_INDEX_SAFE
#undef OUTPUT_GET_INDEX_SAFE
#endif
#ifdef OUTPUT_GET_INDEX
#undef OUTPUT_GET_INDEX
#endif
#ifdef OUTPUT_GET_INDEX_RAW
#undef OUTPUT_GET_INDEX_RAW
#endif
#ifdef OUTPUT_VIEW_OFFSET
#undef OUTPUT_VIEW_OFFSET
#endif
#ifdef OUTPUT_DIMS
#undef OUTPUT_DIMS
#endif
#ifdef OUTPUT_SIMPLE
#undef OUTPUT_SIMPLE
#endif
#ifdef OUTPUT_GROUPED
#undef OUTPUT_GROUPED
#endif
#ifdef OUTPUT_LAYOUT_BFYX
#undef OUTPUT_LAYOUT_BFYX
#endif
#ifdef OUTPUT_TYPE
#undef OUTPUT_TYPE
#endif
#ifdef OUTPUT_VAL_MAX
#undef OUTPUT_VAL_MAX
#endif
#ifdef OUTPUT_VAL_MIN
#undef OUTPUT_VAL_MIN
#endif
#ifdef OUTPUT_VAL_ONE
#undef OUTPUT_VAL_ONE
#endif
#ifdef OUTPUT_VAL_ZERO
#undef OUTPUT_VAL_ZERO
#endif
#ifdef TO_OUTPUT_TYPE
#undef TO_OUTPUT_TYPE
#endif
#ifdef TO_OUTPUT_TYPE_SAT
#undef TO_OUTPUT_TYPE_SAT
#endif
#ifdef AS_OUTPUT_TYPE
#undef AS_OUTPUT_TYPE
#endif
#ifdef OUTPUT_MAX_FUNC
#undef OUTPUT_MAX_FUNC
#endif
#ifdef OUTPUT_MIN_FUNC
#undef OUTPUT_MIN_FUNC
#endif
#ifdef OUTPUT_ABS_FUNC
#undef OUTPUT_ABS_FUNC
#endif
#ifdef OUTPUT_TYPE_SIZE
#undef OUTPUT_TYPE_SIZE
#endif
#ifdef OUTPUT_IS_FP
#undef OUTPUT_IS_FP
#endif
#ifdef OUTPUT_OFFSET
#undef OUTPUT_OFFSET
#endif
#ifdef OUTPUT_SIZES_DATA
#undef OUTPUT_SIZES_DATA
#endif
#ifdef OUTPUT_PITCHES
#undef OUTPUT_PITCHES
#endif
#ifdef OUTPUT_PAD_BEFORE
#undef OUTPUT_PAD_BEFORE
#endif
#ifdef OUTPUT_PAD_AFTER
#undef OUTPUT_PAD_AFTER
#endif
#ifdef OPTIONAL_SHAPE_INFO_ARG
#undef OPTIONAL_SHAPE_INFO_ARG
#endif
#ifdef OPTIONAL_SHAPE_INFO_TENSOR
#undef OPTIONAL_SHAPE_INFO_TENSOR
#endif
#ifdef ELTWISE_LAYOUT_BASED
#undef ELTWISE_LAYOUT_BASED
#endif
#ifdef QUANTIZATION_TERM
#undef QUANTIZATION_TERM
#endif
#ifdef ELTWISE_BROADCAST
#undef ELTWISE_BROADCAST
#endif
#ifdef ACCUMULATOR_TYPE
#undef ACCUMULATOR_TYPE
#endif
#ifdef ACCUMULATOR_VAL_MAX
#undef ACCUMULATOR_VAL_MAX
#endif
#ifdef ACCUMULATOR_VAL_MIN
#undef ACCUMULATOR_VAL_MIN
#endif
#ifdef ACCUMULATOR_VAL_ONE
#undef ACCUMULATOR_VAL_ONE
#endif
#ifdef ACCUMULATOR_VAL_ZERO
#undef ACCUMULATOR_VAL_ZERO
#endif
#ifdef TO_ACCUMULATOR_TYPE
#undef TO_ACCUMULATOR_TYPE
#endif
#ifdef TO_ACCUMULATOR_TYPE_SAT
#undef TO_ACCUMULATOR_TYPE_SAT
#endif
#ifdef AS_ACCUMULATOR_TYPE
#undef AS_ACCUMULATOR_TYPE
#endif
#ifdef ACCUMULATOR_MAX_FUNC
#undef ACCUMULATOR_MAX_FUNC
#endif
#ifdef ACCUMULATOR_MIN_FUNC
#undef ACCUMULATOR_MIN_FUNC
#endif
#ifdef ACCUMULATOR_ABS_FUNC
#undef ACCUMULATOR_ABS_FUNC
#endif
#ifdef ACCUMULATOR_TYPE_SIZE
#undef ACCUMULATOR_TYPE_SIZE
#endif
#ifdef ACCUMULATOR_IS_FP
#undef ACCUMULATOR_IS_FP
#endif
#ifdef ELTWISE_NO_PITCH_SAME_DIMS
#undef ELTWISE_NO_PITCH_SAME_DIMS
#endif
#ifdef INPUTS_DECLS
#undef INPUTS_DECLS
#endif
#ifdef OUTPUT_IDX_ORDER
#undef OUTPUT_IDX_ORDER
#endif
#ifdef INPUT0_IDX_ORDER
#undef INPUT0_IDX_ORDER
#endif
#ifdef INPUT1_IDX_ORDER
#undef INPUT1_IDX_ORDER
#endif
#ifdef INPUT_0_0
#undef INPUT_0_0
#endif
#ifdef INPUT_0_1
#undef INPUT_0_1
#endif
#ifdef OPERATION0
#undef OPERATION0
#endif
#ifdef DO_ELTWISE
#undef DO_ELTWISE
#endif
#ifdef GWS_BATCH
#undef GWS_BATCH
#endif
#ifdef GWS_FEATURE
#undef GWS_FEATURE
#endif
#ifdef GWS_YX
#undef GWS_YX
#endif
#ifdef NL_M_TYPED
#undef NL_M_TYPED
#endif
#ifdef NL_N_TYPED
#undef NL_N_TYPED
#endif
#ifdef ACTIVATION_FUNC_TYPED_TYPE
#undef ACTIVATION_FUNC_TYPED_TYPE
#endif
#ifdef ACTIVATION_FUNC_TYPED_VAL_MAX
#undef ACTIVATION_FUNC_TYPED_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_TYPED_VAL_MIN
#undef ACTIVATION_FUNC_TYPED_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_TYPED_VAL_ONE
#undef ACTIVATION_FUNC_TYPED_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_TYPED_VAL_ZERO
#undef ACTIVATION_FUNC_TYPED_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_TYPED_TYPE
#undef TO_ACTIVATION_FUNC_TYPED_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_TYPED_TYPE_SAT
#undef TO_ACTIVATION_FUNC_TYPED_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_TYPED_TYPE
#undef AS_ACTIVATION_FUNC_TYPED_TYPE
#endif
#ifdef ACTIVATION_FUNC_TYPED_MAX_FUNC
#undef ACTIVATION_FUNC_TYPED_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_TYPED_MIN_FUNC
#undef ACTIVATION_FUNC_TYPED_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_TYPED_ABS_FUNC
#undef ACTIVATION_FUNC_TYPED_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_TYPED_TYPE_SIZE
#undef ACTIVATION_FUNC_TYPED_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_TYPED_IS_FP
#undef ACTIVATION_FUNC_TYPED_IS_FP
#endif
#ifdef ACTIVATION_PARAMS_TYPED
#undef ACTIVATION_PARAMS_TYPED
#endif
#ifdef ACTIVATION_FUNC_TYPED
#undef ACTIVATION_FUNC_TYPED
#endif
#ifdef ACTIVATION_TYPED
#undef ACTIVATION_TYPED
#endif
#ifdef FUSED_OP0_LOAD
#undef FUSED_OP0_LOAD
#endif
#ifdef FUSED_OP0_ACTION
#undef FUSED_OP0_ACTION
#endif
#ifdef FUSED_OPS
#undef FUSED_OPS
#endif
#ifdef FUSED_OPS_PRELOAD
#undef FUSED_OPS_PRELOAD
#endif
#ifdef FUSED_OPS_CALC
#undef FUSED_OPS_CALC
#endif
#ifdef FUSED_OPS_RESULT
#undef FUSED_OPS_RESULT
#endif
#ifdef FUSED_OPS_CAN_USE_PRELOAD
#undef FUSED_OPS_CAN_USE_PRELOAD
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_X
#undef FUSED_OP_0_INPUT0_SIZE_X
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_Y
#undef FUSED_OP_0_INPUT0_SIZE_Y
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_Z
#undef FUSED_OP_0_INPUT0_SIZE_Z
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_W
#undef FUSED_OP_0_INPUT0_SIZE_W
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_U
#undef FUSED_OP_0_INPUT0_SIZE_U
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_V
#undef FUSED_OP_0_INPUT0_SIZE_V
#endif
#ifdef FUSED_OP_0_INPUT0_FEATURE_NUM
#undef FUSED_OP_0_INPUT0_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_BATCH_NUM
#undef FUSED_OP_0_INPUT0_BATCH_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_X
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_X
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Y
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Y
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Z
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Z
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_W
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_W
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_U
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_U
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_V
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_V
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_FEATURE_NUM
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_BATCH_NUM
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_BATCH_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_X
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_X
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Y
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Y
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Z
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Z
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_W
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_W
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_U
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_U
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_V
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_V
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_FEATURE_NUM
#undef FUSED_OP_0_INPUT0_PAD_AFTER_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_BATCH_NUM
#undef FUSED_OP_0_INPUT0_PAD_AFTER_BATCH_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_LENGTH
#undef FUSED_OP_0_INPUT0_LENGTH
#endif
#ifdef FUSED_OP_0_INPUT0_X_PITCH
#undef FUSED_OP_0_INPUT0_X_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_Y_PITCH
#undef FUSED_OP_0_INPUT0_Y_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_Z_PITCH
#undef FUSED_OP_0_INPUT0_Z_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_W_PITCH
#undef FUSED_OP_0_INPUT0_W_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_U_PITCH
#undef FUSED_OP_0_INPUT0_U_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_V_PITCH
#undef FUSED_OP_0_INPUT0_V_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_FEATURE_PITCH
#undef FUSED_OP_0_INPUT0_FEATURE_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_BATCH_PITCH
#undef FUSED_OP_0_INPUT0_BATCH_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_GET_INDEX_SAFE
#undef FUSED_OP_0_INPUT0_GET_INDEX_SAFE
#endif
#ifdef FUSED_OP_0_INPUT0_GET_INDEX
#undef FUSED_OP_0_INPUT0_GET_INDEX
#endif
#ifdef FUSED_OP_0_INPUT0_GET_INDEX_RAW
#undef FUSED_OP_0_INPUT0_GET_INDEX_RAW
#endif
#ifdef FUSED_OP_0_INPUT0_VIEW_OFFSET
#undef FUSED_OP_0_INPUT0_VIEW_OFFSET
#endif
#ifdef FUSED_OP_0_INPUT0_DIMS
#undef FUSED_OP_0_INPUT0_DIMS
#endif
#ifdef FUSED_OP_0_INPUT0_SIMPLE
#undef FUSED_OP_0_INPUT0_SIMPLE
#endif
#ifdef FUSED_OP_0_INPUT0_GROUPED
#undef FUSED_OP_0_INPUT0_GROUPED
#endif
#ifdef FUSED_OP_0_INPUT0_LAYOUT_BFYX
#undef FUSED_OP_0_INPUT0_LAYOUT_BFYX
#endif
#ifdef FUSED_OP_0_INPUT0_TYPE
#undef FUSED_OP_0_INPUT0_TYPE
#endif
#ifdef FUSED_OP_0_INPUT0_VAL_MAX
#undef FUSED_OP_0_INPUT0_VAL_MAX
#endif
#ifdef FUSED_OP_0_INPUT0_VAL_MIN
#undef FUSED_OP_0_INPUT0_VAL_MIN
#endif
#ifdef FUSED_OP_0_INPUT0_VAL_ONE
#undef FUSED_OP_0_INPUT0_VAL_ONE
#endif
#ifdef FUSED_OP_0_INPUT0_VAL_ZERO
#undef FUSED_OP_0_INPUT0_VAL_ZERO
#endif
#ifdef TO_FUSED_OP_0_INPUT0_TYPE
#undef TO_FUSED_OP_0_INPUT0_TYPE
#endif
#ifdef TO_FUSED_OP_0_INPUT0_TYPE_SAT
#undef TO_FUSED_OP_0_INPUT0_TYPE_SAT
#endif
#ifdef AS_FUSED_OP_0_INPUT0_TYPE
#undef AS_FUSED_OP_0_INPUT0_TYPE
#endif
#ifdef FUSED_OP_0_INPUT0_MAX_FUNC
#undef FUSED_OP_0_INPUT0_MAX_FUNC
#endif
#ifdef FUSED_OP_0_INPUT0_MIN_FUNC
#undef FUSED_OP_0_INPUT0_MIN_FUNC
#endif
#ifdef FUSED_OP_0_INPUT0_ABS_FUNC
#undef FUSED_OP_0_INPUT0_ABS_FUNC
#endif
#ifdef FUSED_OP_0_INPUT0_TYPE_SIZE
#undef FUSED_OP_0_INPUT0_TYPE_SIZE
#endif
#ifdef FUSED_OP_0_INPUT0_IS_FP
#undef FUSED_OP_0_INPUT0_IS_FP
#endif
#ifdef FUSED_OP_0_INPUT0_OFFSET
#undef FUSED_OP_0_INPUT0_OFFSET
#endif
#ifdef FUSED_OP_0_INPUT0_SIZES_DATA
#undef FUSED_OP_0_INPUT0_SIZES_DATA
#endif
#ifdef FUSED_OP_0_INPUT0_PITCHES
#undef FUSED_OP_0_INPUT0_PITCHES
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE
#undef FUSED_OP_0_INPUT0_PAD_BEFORE
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER
#undef FUSED_OP_0_INPUT0_PAD_AFTER
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_X
#undef FUSED_OP_0_OUTPUT_SIZE_X
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_Y
#undef FUSED_OP_0_OUTPUT_SIZE_Y
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_Z
#undef FUSED_OP_0_OUTPUT_SIZE_Z
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_W
#undef FUSED_OP_0_OUTPUT_SIZE_W
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_U
#undef FUSED_OP_0_OUTPUT_SIZE_U
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_V
#undef FUSED_OP_0_OUTPUT_SIZE_V
#endif
#ifdef FUSED_OP_0_OUTPUT_FEATURE_NUM
#undef FUSED_OP_0_OUTPUT_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_BATCH_NUM
#undef FUSED_OP_0_OUTPUT_BATCH_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_X
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_X
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Y
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Y
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Z
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Z
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_W
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_W
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_U
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_U
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_V
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_V
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_FEATURE_NUM
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_BATCH_NUM
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_BATCH_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_X
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_X
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Y
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Y
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Z
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Z
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_W
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_W
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_U
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_U
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_V
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_V
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_FEATURE_NUM
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_BATCH_NUM
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_BATCH_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_LENGTH
#undef FUSED_OP_0_OUTPUT_LENGTH
#endif
#ifdef FUSED_OP_0_OUTPUT_X_PITCH
#undef FUSED_OP_0_OUTPUT_X_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_Y_PITCH
#undef FUSED_OP_0_OUTPUT_Y_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_Z_PITCH
#undef FUSED_OP_0_OUTPUT_Z_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_W_PITCH
#undef FUSED_OP_0_OUTPUT_W_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_U_PITCH
#undef FUSED_OP_0_OUTPUT_U_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_V_PITCH
#undef FUSED_OP_0_OUTPUT_V_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_FEATURE_PITCH
#undef FUSED_OP_0_OUTPUT_FEATURE_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_BATCH_PITCH
#undef FUSED_OP_0_OUTPUT_BATCH_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_GET_INDEX_SAFE
#undef FUSED_OP_0_OUTPUT_GET_INDEX_SAFE
#endif
#ifdef FUSED_OP_0_OUTPUT_GET_INDEX
#undef FUSED_OP_0_OUTPUT_GET_INDEX
#endif
#ifdef FUSED_OP_0_OUTPUT_GET_INDEX_RAW
#undef FUSED_OP_0_OUTPUT_GET_INDEX_RAW
#endif
#ifdef FUSED_OP_0_OUTPUT_VIEW_OFFSET
#undef FUSED_OP_0_OUTPUT_VIEW_OFFSET
#endif
#ifdef FUSED_OP_0_OUTPUT_DIMS
#undef FUSED_OP_0_OUTPUT_DIMS
#endif
#ifdef FUSED_OP_0_OUTPUT_SIMPLE
#undef FUSED_OP_0_OUTPUT_SIMPLE
#endif
#ifdef FUSED_OP_0_OUTPUT_GROUPED
#undef FUSED_OP_0_OUTPUT_GROUPED
#endif
#ifdef FUSED_OP_0_OUTPUT_LAYOUT_BFYX
#undef FUSED_OP_0_OUTPUT_LAYOUT_BFYX
#endif
#ifdef FUSED_OP_0_OUTPUT_TYPE
#undef FUSED_OP_0_OUTPUT_TYPE
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_MAX
#undef FUSED_OP_0_OUTPUT_VAL_MAX
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_MIN
#undef FUSED_OP_0_OUTPUT_VAL_MIN
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_ONE
#undef FUSED_OP_0_OUTPUT_VAL_ONE
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_ZERO
#undef FUSED_OP_0_OUTPUT_VAL_ZERO
#endif
#ifdef TO_FUSED_OP_0_OUTPUT_TYPE
#undef TO_FUSED_OP_0_OUTPUT_TYPE
#endif
#ifdef TO_FUSED_OP_0_OUTPUT_TYPE_SAT
#undef TO_FUSED_OP_0_OUTPUT_TYPE_SAT
#endif
#ifdef AS_FUSED_OP_0_OUTPUT_TYPE
#undef AS_FUSED_OP_0_OUTPUT_TYPE
#endif
#ifdef FUSED_OP_0_OUTPUT_MAX_FUNC
#undef FUSED_OP_0_OUTPUT_MAX_FUNC
#endif
#ifdef FUSED_OP_0_OUTPUT_MIN_FUNC
#undef FUSED_OP_0_OUTPUT_MIN_FUNC
#endif
#ifdef FUSED_OP_0_OUTPUT_ABS_FUNC
#undef FUSED_OP_0_OUTPUT_ABS_FUNC
#endif
#ifdef FUSED_OP_0_OUTPUT_TYPE_SIZE
#undef FUSED_OP_0_OUTPUT_TYPE_SIZE
#endif
#ifdef FUSED_OP_0_OUTPUT_IS_FP
#undef FUSED_OP_0_OUTPUT_IS_FP
#endif
#ifdef FUSED_OP_0_OUTPUT_OFFSET
#undef FUSED_OP_0_OUTPUT_OFFSET
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZES_DATA
#undef FUSED_OP_0_OUTPUT_SIZES_DATA
#endif
#ifdef FUSED_OP_0_OUTPUT_PITCHES
#undef FUSED_OP_0_OUTPUT_PITCHES
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER
#undef FUSED_OP_0_OUTPUT_PAD_AFTER
#endif
#ifdef FUSED_OP0_DECLS
#undef FUSED_OP0_DECLS
#endif
#ifdef FUSED_OP0_ARGS
#undef FUSED_OP0_ARGS
#endif
#ifdef FUSED_OPS_DECLS
#undef FUSED_OPS_DECLS
#endif
#ifdef FUSED_OPS_ARGS
#undef FUSED_OPS_ARGS
#endif
#ifdef HAS_FUSED_OPS
#undef HAS_FUSED_OPS
#endif
#ifdef HAS_FUSED_OPS_DECLS
#undef HAS_FUSED_OPS_DECLS
#endif

//====================================================
// Kernel template: convolution_gpu_bfyx_f16 
// Kernel name: convolution_gpu_bfyx_f16_4065784441082281624_0_0
#define KERNEL(name) __kernel void convolution_gpu_bfyx_f16_4065784441082281624_0_0
#define KERNEL_ID convolution_gpu_bfyx_f16_4065784441082281624_0_0
#define FUNC(name)  _##name##_convolution_gpu_bfyx_f16_4065784441082281624_0_0
#define FUNC_CALL(name)  _##name##_convolution_gpu_bfyx_f16_4065784441082281624_0_0
#define CONST_ARRAY_DECL(name) __constant size_t  _##name##_convolution_gpu_bfyx_f16_4065784441082281624_0_0 []
#define CONST_ARRAY_REF(name)  _##name##_convolution_gpu_bfyx_f16_4065784441082281624_0_0
#define FP64_SUPPORTED 0
#define FP16_SUPPORTED 1
#define FP16_UNIT_USED 1
#define INT8_UNIT_USED 0
#define INT32_UNIT_USED 0
#define INT64_UNIT_USED 0
#define UINT8_UNIT_USED 0
#define UINT32_UNIT_USED 0
#define UNIT_TYPE half
#define UNIT_VAL_MAX HALF_MAX
#define UNIT_VAL_MIN -UNIT_VAL_MAX
#define UNIT_VAL_ONE 1.0h
#define UNIT_VAL_ZERO 0.0h
#define TO_UNIT_TYPE(v) convert_half(v)
#define TO_UNIT_TYPE_SAT(v) convert_half(v)
#define AS_UNIT_TYPE(v) as_half(v)
#define UNIT_MAX_FUNC fmax
#define UNIT_MIN_FUNC fmin
#define UNIT_ABS_FUNC fabs
#define UNIT_TYPE_SIZE 2
#define UNIT_IS_FP 1
#define NL_M as_float(0x0)/*0.000000e+00*/
#define NL_N as_float(0x0)/*0.000000e+00*/
#define ACTIVATION_FUNC_TYPE half
#define ACTIVATION_FUNC_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_VAL_MIN -ACTIVATION_FUNC_VAL_MAX
#define ACTIVATION_FUNC_VAL_ONE 1.0h
#define ACTIVATION_FUNC_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_MAX_FUNC fmax
#define ACTIVATION_FUNC_MIN_FUNC fmin
#define ACTIVATION_FUNC_ABS_FUNC fabs
#define ACTIVATION_FUNC_TYPE_SIZE 2
#define ACTIVATION_FUNC_IS_FP 1
#define ACTIVATION_PARAMS NL_M, NL_N
#define ACTIVATION_FUNC(input, m, n) input
#define ACTIVATION(input, params) ACTIVATION_FUNC(input, params)
#define INPUT0_SIZE_X 7
#define INPUT0_SIZE_Y 7
#define INPUT0_SIZE_Z 1
#define INPUT0_SIZE_W 1
#define INPUT0_SIZE_U 1
#define INPUT0_SIZE_V 1
#define INPUT0_FEATURE_NUM 512
#define INPUT0_BATCH_NUM 1
#define INPUT0_PAD_BEFORE_SIZE_X 1
#define INPUT0_PAD_BEFORE_SIZE_Y 1
#define INPUT0_PAD_BEFORE_SIZE_Z 0
#define INPUT0_PAD_BEFORE_SIZE_W 0
#define INPUT0_PAD_BEFORE_SIZE_U 0
#define INPUT0_PAD_BEFORE_SIZE_V 0
#define INPUT0_PAD_BEFORE_FEATURE_NUM 0
#define INPUT0_PAD_BEFORE_BATCH_NUM 0
#define INPUT0_PAD_AFTER_SIZE_X 1
#define INPUT0_PAD_AFTER_SIZE_Y 1
#define INPUT0_PAD_AFTER_SIZE_Z 0
#define INPUT0_PAD_AFTER_SIZE_W 0
#define INPUT0_PAD_AFTER_SIZE_U 0
#define INPUT0_PAD_AFTER_SIZE_V 0
#define INPUT0_PAD_AFTER_FEATURE_NUM 0
#define INPUT0_PAD_AFTER_BATCH_NUM 0
#define INPUT0_LENGTH 25088
#define INPUT0_X_PITCH 1
#define INPUT0_Y_PITCH 9
#define INPUT0_Z_PITCH 1
#define INPUT0_W_PITCH 1
#define INPUT0_U_PITCH 1
#define INPUT0_V_PITCH 1
#define INPUT0_FEATURE_PITCH 81
#define INPUT0_BATCH_PITCH 41472
#define INPUT0_GET_INDEX_SAFE(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX_SAFE(INPUT0, b, f, y, x)
#define INPUT0_GET_INDEX(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(INPUT0, b, f, y, x)
#define INPUT0_GET_INDEX_RAW(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(INPUT0, b, f, y, x)
#define INPUT0_VIEW_OFFSET 0
#define INPUT0_DIMS 4
#define INPUT0_SIMPLE 0
#define INPUT0_GROUPED 0
#define INPUT0_LAYOUT_B_FS_YX_FSV16 1
#define INPUT0_TYPE half
#define INPUT0_VAL_MAX HALF_MAX
#define INPUT0_VAL_MIN -INPUT0_VAL_MAX
#define INPUT0_VAL_ONE 1.0h
#define INPUT0_VAL_ZERO 0.0h
#define TO_INPUT0_TYPE(v) convert_half(v)
#define TO_INPUT0_TYPE_SAT(v) convert_half(v)
#define AS_INPUT0_TYPE(v) as_half(v)
#define INPUT0_MAX_FUNC fmax
#define INPUT0_MIN_FUNC fmin
#define INPUT0_ABS_FUNC fabs
#define INPUT0_TYPE_SIZE 2
#define INPUT0_IS_FP 1
#define INPUT0_OFFSET 10
#define INPUT0_SIZES_DATA { 7,7,512,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(INPUT0_SIZES) = INPUT0_SIZES_DATA;
#define INPUT0_SIZES CONST_ARRAY_REF(INPUT0_SIZES)
#define INPUT0_PITCHES (size_t []){ 1,9,81,41472,1,1,1,1,1, } 
#define INPUT0_PAD_BEFORE (size_t []){ 1,1,0,0,0,0,0,0,0, } 
#define INPUT0_PAD_AFTER (size_t []){ 1,1,0,0,0,0,0,0,0, } 
#define OUTPUT_SIZE_X 7
#define OUTPUT_SIZE_Y 7
#define OUTPUT_SIZE_Z 1
#define OUTPUT_SIZE_W 1
#define OUTPUT_SIZE_U 1
#define OUTPUT_SIZE_V 1
#define OUTPUT_FEATURE_NUM 512
#define OUTPUT_BATCH_NUM 1
#define OUTPUT_PAD_BEFORE_SIZE_X 0
#define OUTPUT_PAD_BEFORE_SIZE_Y 0
#define OUTPUT_PAD_BEFORE_SIZE_Z 0
#define OUTPUT_PAD_BEFORE_SIZE_W 0
#define OUTPUT_PAD_BEFORE_SIZE_U 0
#define OUTPUT_PAD_BEFORE_SIZE_V 0
#define OUTPUT_PAD_BEFORE_FEATURE_NUM 0
#define OUTPUT_PAD_BEFORE_BATCH_NUM 0
#define OUTPUT_PAD_AFTER_SIZE_X 0
#define OUTPUT_PAD_AFTER_SIZE_Y 0
#define OUTPUT_PAD_AFTER_SIZE_Z 0
#define OUTPUT_PAD_AFTER_SIZE_W 0
#define OUTPUT_PAD_AFTER_SIZE_U 0
#define OUTPUT_PAD_AFTER_SIZE_V 0
#define OUTPUT_PAD_AFTER_FEATURE_NUM 0
#define OUTPUT_PAD_AFTER_BATCH_NUM 0
#define OUTPUT_LENGTH 25088
#define OUTPUT_X_PITCH 1
#define OUTPUT_Y_PITCH 7
#define OUTPUT_Z_PITCH 1
#define OUTPUT_W_PITCH 1
#define OUTPUT_U_PITCH 1
#define OUTPUT_V_PITCH 1
#define OUTPUT_FEATURE_PITCH 49
#define OUTPUT_BATCH_PITCH 25088
#define OUTPUT_GET_INDEX_SAFE(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX_SAFE(OUTPUT, b, f, y, x)
#define OUTPUT_GET_INDEX(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(OUTPUT, b, f, y, x)
#define OUTPUT_GET_INDEX_RAW(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(OUTPUT, b, f, y, x)
#define OUTPUT_VIEW_OFFSET 0
#define OUTPUT_DIMS 4
#define OUTPUT_SIMPLE 0
#define OUTPUT_GROUPED 0
#define OUTPUT_LAYOUT_B_FS_YX_FSV16 1
#define OUTPUT_TYPE half
#define OUTPUT_VAL_MAX HALF_MAX
#define OUTPUT_VAL_MIN -OUTPUT_VAL_MAX
#define OUTPUT_VAL_ONE 1.0h
#define OUTPUT_VAL_ZERO 0.0h
#define TO_OUTPUT_TYPE(v) convert_half(v)
#define TO_OUTPUT_TYPE_SAT(v) convert_half(v)
#define AS_OUTPUT_TYPE(v) as_half(v)
#define OUTPUT_MAX_FUNC fmax
#define OUTPUT_MIN_FUNC fmin
#define OUTPUT_ABS_FUNC fabs
#define OUTPUT_TYPE_SIZE 2
#define OUTPUT_IS_FP 1
#define OUTPUT_OFFSET 0
#define OUTPUT_SIZES_DATA { 7,7,512,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(OUTPUT_SIZES) = OUTPUT_SIZES_DATA;
#define OUTPUT_SIZES CONST_ARRAY_REF(OUTPUT_SIZES)
#define OUTPUT_PITCHES (size_t []){ 1,7,49,25088,1,1,1,1,1, } 
#define OUTPUT_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OUTPUT_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OPTIONAL_SHAPE_INFO_ARG 
#define OPTIONAL_SHAPE_INFO_TENSOR 
#define FILTER_SIZE_X 3
#define FILTER_SIZE_Y 3
#define FILTER_SIZE_Z 1
#define FILTER_IFM_NUM 512
#define FILTER_OFM_NUM 512
#define FILTER_GROUPS_NUM 1
#define FILTER_X_PITCH 1
#define FILTER_Y_PITCH 3
#define FILTER_Z_PITCH 1
#define FILTER_IFM_PITCH 9
#define FILTER_OFM_PITCH 4608
#define FILTER_GROUPS_PITCH 1
#define FILTER_LENGTH 2359296
#define FILTER_VIEW_OFFSET 0
#define FILTER_DIMS 4
#define FILTER_SIMPLE 0
#define FILTER_GROUPED 0
#define FILTER_LAYOUT_OS_IS_YX_ISV16_OSV16 1
#define FILTER_TYPE half
#define FILTER_VAL_MAX HALF_MAX
#define FILTER_VAL_MIN -FILTER_VAL_MAX
#define FILTER_VAL_ONE 1.0h
#define FILTER_VAL_ZERO 0.0h
#define TO_FILTER_TYPE(v) convert_half(v)
#define TO_FILTER_TYPE_SAT(v) convert_half(v)
#define AS_FILTER_TYPE(v) as_half(v)
#define FILTER_MAX_FUNC fmax
#define FILTER_MIN_FUNC fmin
#define FILTER_ABS_FUNC fabs
#define FILTER_TYPE_SIZE 2
#define FILTER_IS_FP 1
#define FILTER_OFFSET 0
#define FILTER_SIZES_DATA { 3,3,512,512,1,1,1,1,1, } 
CONST_ARRAY_DECL(FILTER_SIZES) = FILTER_SIZES_DATA;
#define FILTER_SIZES CONST_ARRAY_REF(FILTER_SIZES)
#define FILTER_PITCHES (size_t []){ 1,3,9,4608,1,1,1,1,1, } 
#define FILTER_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FILTER_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define BIAS_TERM 1
#define BIAS_SIZE_X 1
#define BIAS_SIZE_Y 1
#define BIAS_SIZE_Z 1
#define BIAS_SIZE_W 1
#define BIAS_SIZE_U 1
#define BIAS_SIZE_V 1
#define BIAS_FEATURE_NUM 512
#define BIAS_BATCH_NUM 1
#define BIAS_PAD_BEFORE_SIZE_X 0
#define BIAS_PAD_BEFORE_SIZE_Y 0
#define BIAS_PAD_BEFORE_SIZE_Z 0
#define BIAS_PAD_BEFORE_SIZE_W 0
#define BIAS_PAD_BEFORE_SIZE_U 0
#define BIAS_PAD_BEFORE_SIZE_V 0
#define BIAS_PAD_BEFORE_FEATURE_NUM 0
#define BIAS_PAD_BEFORE_BATCH_NUM 0
#define BIAS_PAD_AFTER_SIZE_X 0
#define BIAS_PAD_AFTER_SIZE_Y 0
#define BIAS_PAD_AFTER_SIZE_Z 0
#define BIAS_PAD_AFTER_SIZE_W 0
#define BIAS_PAD_AFTER_SIZE_U 0
#define BIAS_PAD_AFTER_SIZE_V 0
#define BIAS_PAD_AFTER_FEATURE_NUM 0
#define BIAS_PAD_AFTER_BATCH_NUM 0
#define BIAS_LENGTH 512
#define BIAS_X_PITCH 1
#define BIAS_Y_PITCH 1
#define BIAS_Z_PITCH 1
#define BIAS_W_PITCH 1
#define BIAS_U_PITCH 1
#define BIAS_V_PITCH 1
#define BIAS_FEATURE_PITCH 1
#define BIAS_BATCH_PITCH 512
#define BIAS_GET_INDEX_SAFE(b, f, y, x) ((0 + (f)) % 512)
#define BIAS_GET_INDEX(b, f, y, x) (0 + (f))
#define BIAS_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(BIAS, b, f, y, x)
#define BIAS_VIEW_OFFSET 0
#define BIAS_DIMS 2
#define BIAS_SIMPLE 1
#define BIAS_GROUPED 0
#define BIAS_LAYOUT_BF 1
#define BIAS_TYPE half
#define BIAS_VAL_MAX HALF_MAX
#define BIAS_VAL_MIN -BIAS_VAL_MAX
#define BIAS_VAL_ONE 1.0h
#define BIAS_VAL_ZERO 0.0h
#define TO_BIAS_TYPE(v) convert_half(v)
#define TO_BIAS_TYPE_SAT(v) convert_half(v)
#define AS_BIAS_TYPE(v) as_half(v)
#define BIAS_MAX_FUNC fmax
#define BIAS_MIN_FUNC fmin
#define BIAS_ABS_FUNC fabs
#define BIAS_TYPE_SIZE 2
#define BIAS_IS_FP 1
#define BIAS_OFFSET 0
#define BIAS_SIZES_DATA { 512,1,1,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(BIAS_SIZES) = BIAS_SIZES_DATA;
#define BIAS_SIZES CONST_ARRAY_REF(BIAS_SIZES)
#define BIAS_PITCHES (size_t []){ 1,512,1,1,1,1,1,1,1, } 
#define BIAS_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define BIAS_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define BIAS_PER_OUTPUT 0
#define BIAS_PER_OFM 1
#define STRIDE_SIZE_X 1
#define STRIDE_SIZE_Y 1
#define STRIDE_SIZE_Z 1
#define PADDING_SIZE_X 1
#define PADDING_SIZE_Y 1
#define PADDING_SIZE_Z 0
#define DILATION_SIZE_X 1
#define DILATION_SIZE_Y 1
#define DILATION_SIZE_Z 1
#define FILTER_ARRAY_NUM 1
#define INPUT0_OFFSET_WITH_PADDING 0
#define GROUPED 0
#define SKIP_BATCH 1
#define FUSED_OP0_LOAD_VEC 
#define ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE half
#define ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_MIN -ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_MAX
#define ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_ONE 1.0h
#define ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_FUSED_OP0_VEC_MAX_FUNC fmax
#define ACTIVATION_FUNC_FUSED_OP0_VEC_MIN_FUNC fmin
#define ACTIVATION_FUNC_FUSED_OP0_VEC_ABS_FUNC fabs
#define ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE_SIZE 2
#define ACTIVATION_FUNC_FUSED_OP0_VEC_IS_FP 1
#define ACTIVATION_PARAMS_FUSED_OP0_VEC NL_M_FUSED_OP0_VEC, NL_N_FUSED_OP0_VEC
#define ACTIVATION_FUNC_FUSED_OP0_VEC(input, m, n) (ACTIVATION_FUNC_FUSED_OP0_VEC_MAX_FUNC(ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_ZERO, input))
#define ACTIVATION_FUSED_OP0_VEC(input, params) ACTIVATION_FUNC_FUSED_OP0_VEC(input, params)
#define FUSED_OP0_ACTION_VEC \
	half8 dst_out_0 = convert_half8(dst);\
	dst_out_0 = ACTIVATION_FUNC_FUSED_OP0_VEC(dst_out_0, (half8)(as_float(0x0)/*0.000000e+00*/),(half8)(as_float(0x0)/*0.000000e+00*/));
#define FUSED_OPS_VEC \
	FUSED_OP0_LOAD_VEC\
	FUSED_OP0_ACTION_VEC
#define FUSED_OPS_PRELOAD_VEC \
	FUSED_OP0_LOAD_VEC
#define FUSED_OPS_CALC_VEC \
	FUSED_OP0_ACTION_VEC
#define FUSED_OPS_RESULT_VEC dst_out_0
#define FUSED_OPS_CAN_USE_PRELOAD_VEC 1
#define FUSED_OP0_LOAD_SCALAR 
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE half
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_MIN -ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_MAX
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_ONE 1.0h
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_MAX_FUNC fmax
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_MIN_FUNC fmin
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_ABS_FUNC fabs
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE_SIZE 2
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_IS_FP 1
#define ACTIVATION_PARAMS_FUSED_OP0_SCALAR NL_M_FUSED_OP0_SCALAR, NL_N_FUSED_OP0_SCALAR
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR(input, m, n) (ACTIVATION_FUNC_FUSED_OP0_SCALAR_MAX_FUNC(ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_ZERO, input))
#define ACTIVATION_FUSED_OP0_SCALAR(input, params) ACTIVATION_FUNC_FUSED_OP0_SCALAR(input, params)
#define FUSED_OP0_ACTION_SCALAR \
	half dst_i__out_0 = convert_half(dst[i]);\
	dst_i__out_0 = ACTIVATION_FUNC_FUSED_OP0_SCALAR(dst_i__out_0, (half)(as_float(0x0)/*0.000000e+00*/),(half)(as_float(0x0)/*0.000000e+00*/));
#define FUSED_OPS_SCALAR \
	FUSED_OP0_LOAD_SCALAR\
	FUSED_OP0_ACTION_SCALAR
#define FUSED_OPS_PRELOAD_SCALAR \
	FUSED_OP0_LOAD_SCALAR
#define FUSED_OPS_CALC_SCALAR \
	FUSED_OP0_ACTION_SCALAR
#define FUSED_OPS_RESULT_SCALAR dst_i__out_0
#define FUSED_OPS_CAN_USE_PRELOAD_SCALAR 1
#define FUSED_OP_0_OUTPUT_SIZE_X 7
#define FUSED_OP_0_OUTPUT_SIZE_Y 7
#define FUSED_OP_0_OUTPUT_SIZE_Z 1
#define FUSED_OP_0_OUTPUT_SIZE_W 1
#define FUSED_OP_0_OUTPUT_SIZE_U 1
#define FUSED_OP_0_OUTPUT_SIZE_V 1
#define FUSED_OP_0_OUTPUT_FEATURE_NUM 512
#define FUSED_OP_0_OUTPUT_BATCH_NUM 1
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_X 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Y 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Z 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_W 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_U 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_V 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_FEATURE_NUM 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_BATCH_NUM 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_X 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Y 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Z 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_W 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_U 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_V 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_FEATURE_NUM 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_BATCH_NUM 0
#define FUSED_OP_0_OUTPUT_LENGTH 25088
#define FUSED_OP_0_OUTPUT_X_PITCH 1
#define FUSED_OP_0_OUTPUT_Y_PITCH 7
#define FUSED_OP_0_OUTPUT_Z_PITCH 1
#define FUSED_OP_0_OUTPUT_W_PITCH 1
#define FUSED_OP_0_OUTPUT_U_PITCH 1
#define FUSED_OP_0_OUTPUT_V_PITCH 1
#define FUSED_OP_0_OUTPUT_FEATURE_PITCH 49
#define FUSED_OP_0_OUTPUT_BATCH_PITCH 25088
#define FUSED_OP_0_OUTPUT_GET_INDEX_SAFE(b, f, y, x) GET_DATA_INDEX_SAFE(FUSED_OP_0_OUTPUT, b, f, y, x)
#define FUSED_OP_0_OUTPUT_GET_INDEX(b, f, y, x) GET_DATA_INDEX(FUSED_OP_0_OUTPUT, b, f, y, x)
#define FUSED_OP_0_OUTPUT_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(FUSED_OP_0_OUTPUT, b, f, y, x)
#define FUSED_OP_0_OUTPUT_VIEW_OFFSET 0
#define FUSED_OP_0_OUTPUT_DIMS 4
#define FUSED_OP_0_OUTPUT_SIMPLE 1
#define FUSED_OP_0_OUTPUT_GROUPED 0
#define FUSED_OP_0_OUTPUT_LAYOUT_BFYX 1
#define FUSED_OP_0_OUTPUT_TYPE half
#define FUSED_OP_0_OUTPUT_VAL_MAX HALF_MAX
#define FUSED_OP_0_OUTPUT_VAL_MIN -FUSED_OP_0_OUTPUT_VAL_MAX
#define FUSED_OP_0_OUTPUT_VAL_ONE 1.0h
#define FUSED_OP_0_OUTPUT_VAL_ZERO 0.0h
#define TO_FUSED_OP_0_OUTPUT_TYPE(v) convert_half(v)
#define TO_FUSED_OP_0_OUTPUT_TYPE_SAT(v) convert_half(v)
#define AS_FUSED_OP_0_OUTPUT_TYPE(v) as_half(v)
#define FUSED_OP_0_OUTPUT_MAX_FUNC fmax
#define FUSED_OP_0_OUTPUT_MIN_FUNC fmin
#define FUSED_OP_0_OUTPUT_ABS_FUNC fabs
#define FUSED_OP_0_OUTPUT_TYPE_SIZE 2
#define FUSED_OP_0_OUTPUT_IS_FP 1
#define FUSED_OP_0_OUTPUT_OFFSET 0
#define FUSED_OP_0_OUTPUT_SIZES_DATA { 7,7,512,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(FUSED_OP_0_OUTPUT_SIZES) = FUSED_OP_0_OUTPUT_SIZES_DATA;
#define FUSED_OP_0_OUTPUT_SIZES CONST_ARRAY_REF(FUSED_OP_0_OUTPUT_SIZES)
#define FUSED_OP_0_OUTPUT_PITCHES (size_t []){ 1,7,49,25088,1,1,1,1,1, } 
#define FUSED_OP_0_OUTPUT_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP_0_OUTPUT_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP0_DECLS 
#define FUSED_OP0_ARGS 
#define FUSED_OPS_DECLS 
#define FUSED_OPS_ARGS 
#define HAS_FUSED_OPS 1
#define HAS_FUSED_OPS_DECLS 0
#define OUTPUT_X_BLOCK_SIZE 8
#define INPUT_LINE_SIZE 9
#define SUB_GROUP_SIZE 16
#define X_BLOCKS 1
#define SLM_DIV_FACTOR 16
#define WORK_GROUP_SIZE 256
#define IC_BLOCKS 32


#define INPUT_TYPE INPUT0_TYPE
#define INPUT_TYPE2 MAKE_VECTOR_TYPE(INPUT0_TYPE, 2)
#define INPUT_TYPE4 MAKE_VECTOR_TYPE(INPUT0_TYPE, 4)
#define INPUT_TYPE8 MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)
#define FILTER_TYPE8 MAKE_VECTOR_TYPE(FILTER_TYPE, 8)
#define AS_INPUT_TYPE CAT(as_, INPUT_TYPE)
#define AS_INPUT_TYPE2 CAT(as_, INPUT_TYPE2)
#define AS_INPUT_TYPE4 CAT(as_, INPUT_TYPE4)
#define AS_INPUT_TYPE8 CAT(as_, INPUT_TYPE8)
#define AS_FILTER_TYPE8 CAT(as_, FILTER_TYPE8)
#if OUTPUT_FORMAT_BFYX
# define OUTPUTVTYPE(n) CAT(OUTPUT_TYPE, n)
# define TO_OUTPUTVTYPE CAT(convert_, OUTPUTVTYPE(OUTPUT_X_BLOCK_SIZE))
# define VSTORE CAT(vstore, OUTPUT_X_BLOCK_SIZE)
#endif
#if INPUT0_TYPE_SIZE == 2
# define AS_INPUT_SRC CAT(as_, MAKE_VECTOR_TYPE(INPUT_TYPE, OUTPUT_X_BLOCK_SIZE))
# define AS_US_SRC CAT(as_, MAKE_VECTOR_TYPE(ushort, OUTPUT_X_BLOCK_SIZE))
# define GET_SRC(data, id) AS_INPUT_SRC(_sub_group_shuffle(AS_US_SRC(data), id))
#else
# define GET_SRC(data, id) _sub_group_shuffle(data, id)
#endif
#define FEATURE_SLICE_SIZE 16
#define FILTER_OFM_NUM_ALIGNED (((FILTER_OFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE) * FEATURE_SLICE_SIZE)
#define FILTER_IFM_NUM_ALIGNED (((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE) * FEATURE_SLICE_SIZE)
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, SUB_GROUP_SIZE * SLM_DIV_FACTOR, 1)))
KERNEL(convolution_bfyx_f16)(
 __global INPUT0_TYPE* input,
 __global OUTPUT_TYPE* output,
 __global FILTER_TYPE* weights
#if BIAS_TERM
 , __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
 , FUSED_OPS_DECLS
#endif
) {
 const int sglid = get_sub_group_local_id();
 const int b = (uint)get_global_id(2);
 const int xy = get_global_id(0);
 const int x = (xy % X_BLOCKS) * OUTPUT_X_BLOCK_SIZE;
 const int y = (xy / X_BLOCKS);
 const int lid1 = (int)get_local_id(1);
 const int feature_per_wg = (int)get_local_size(1) / SLM_DIV_FACTOR;
 const int feature_sub_block = lid1 / feature_per_wg;
 const int feature_block = (int)get_group_id(1);
#if GROUPED
 const int group = (feature_block * FEATURE_SLICE_SIZE) / FILTER_OFM_NUM;
 const int prev_group_leftover = (FILTER_OFM_NUM * (group + 1)) - (feature_block * FEATURE_SLICE_SIZE);
 int groups_per_sub_group = 1;
 if (prev_group_leftover < 16)
 groups_per_sub_group += ((FEATURE_SLICE_SIZE - prev_group_leftover - 1) / FILTER_OFM_NUM) + 1;
#else
 const int group = 0;
 const int groups_per_sub_group = 1;
#endif
 typedef MAKE_VECTOR_TYPE(INPUT0_TYPE, OUTPUT_X_BLOCK_SIZE) vec_t;
 const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
 const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;
 const uint input_x_pitch = FEATURE_SLICE_SIZE;
 const uint input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
 const uint input_fs_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
 const uint input_total_f_size = INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
 const uint input_b_pitch = input_fs_pitch * ((input_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
 const uint input_fs_pad_before = INPUT0_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;
 const uint input_offset = b * input_b_pitch +
 input_fs_pad_before * input_fs_pitch +
 (INPUT0_PAD_BEFORE_SIZE_Y + input_y) * input_y_pitch +
 (INPUT0_PAD_BEFORE_SIZE_X + input_x) * input_x_pitch;
#if OUTPUT_FORMAT_BFYX
 const uint output_y_pitch = (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
 const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
 const uint output_b_pitch = output_fs_pitch * (OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM);
 const uint output_offset = b * output_b_pitch +
 feature_block * (output_fs_pitch * FEATURE_SLICE_SIZE) +
 (sglid + OUTPUT_PAD_BEFORE_FEATURE_NUM) * output_fs_pitch +
 (y + OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
 (x + OUTPUT_PAD_BEFORE_SIZE_X);
#else
 const uint output_x_pitch = FEATURE_SLICE_SIZE;
 const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
 const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
 const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
 const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
 const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;
 const uint output_offset = b * output_b_pitch +
 (feature_block + output_fs_pad_before) * output_fs_pitch +
 (y + OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
 (x + OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;
#endif
 const uint filter_isv_pitch = FEATURE_SLICE_SIZE;
 const uint filter_x_pitch = FEATURE_SLICE_SIZE * FEATURE_SLICE_SIZE;
 const uint filter_y_pitch = filter_x_pitch * FILTER_SIZE_X;
 const uint filter_is_pitch = filter_y_pitch * FILTER_SIZE_Y;
 const uint filter_os_pitch = filter_is_pitch * ((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
#if BIAS_TERM
#if SLM_DIV_FACTOR == 1
 vec_t dst = (vec_t)(DT_INPUT_BLOCK_READ(biases, feature_block * FEATURE_SLICE_SIZE));
#else
 vec_t dst;
 if (feature_sub_block == 0) {
 dst = (vec_t)(DT_INPUT_BLOCK_READ(biases, feature_block * FEATURE_SLICE_SIZE));
 } else {
 dst = INPUT0_VAL_ZERO;
 }
#endif
#else
 vec_t dst = INPUT0_VAL_ZERO;
#endif
#if SLM_DIV_FACTOR > 1
 __local vec_t partial_summ[WORK_GROUP_SIZE];
#endif
#if MULTIPLE_GROUPS_INPUT_PRELOAD
 const uint in_split_offset = feature_block * input_fs_pitch;
 const uint g = sglid / (FEATURE_SLICE_SIZE / groups_per_sub_group);
 const uint ofm_in_group = sglid % (FEATURE_SLICE_SIZE / groups_per_sub_group);
 const uint grouped_filter_offset = (group + g) * FILTER_GROUPS_PITCH;
#else
#if GROUPED
 for (uint g = group; g < group + groups_per_sub_group; g++) {
 const uint in_split_offset = g * input_fs_pitch * (FILTER_IFM_NUM / FEATURE_SLICE_SIZE);
 const uint filter_split_offset = g * FILTER_GROUPS_PITCH;
 const uint filter_offset = (feature_block % (FILTER_OFM_NUM / FEATURE_SLICE_SIZE)) * filter_os_pitch;
#else
 const uint in_split_offset = 0;
 const uint filter_split_offset = 0;
 const uint filter_offset = feature_block * filter_os_pitch;
#endif
 const uint grouped_filter_offset = filter_offset + filter_split_offset;
#endif
 const uint grouped_input_offset = input_offset + in_split_offset;
#if SLM_DIV_FACTOR > 1
 for (int icb = feature_sub_block * IC_BLOCKS / SLM_DIV_FACTOR; icb < (feature_sub_block + 1) * IC_BLOCKS / SLM_DIV_FACTOR; icb++) {
#else
 for (int icb = 0; icb < IC_BLOCKS; icb++) {
#endif
 __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
 for (int kh = 0; kh < FILTER_SIZE_Y; kh++) {
 if (input_y + kh * DILATION_SIZE_Y < 0 || input_y + kh * DILATION_SIZE_Y >= INPUT0_SIZE_Y)
 continue;
 INPUT_TYPE line_cache[INPUT_LINE_SIZE];
#if INPUT_LEFTOVERS
 if ((icb + 1) * FEATURE_SLICE_SIZE >= FILTER_IFM_NUM)
 {
 for (int xb = 0; xb < INPUT_LINE_SIZE; xb++)
 {
 if (icb * FEATURE_SLICE_SIZE + sglid >= FILTER_IFM_NUM)
 line_cache[xb] = 0;
 else
 line_cache[xb] = input[grouped_input_offset +
 icb * input_fs_pitch +
 kh * DILATION_SIZE_Y * input_y_pitch +
 xb * input_x_pitch +
 sglid];
 }
 }
 else
#endif
 {
 int xb = 0;
 for (; xb + 8 <= INPUT_LINE_SIZE; xb += 8) {
 INPUT_TYPE8 vv = DT_INPUT_BLOCK_READ8(input, grouped_input_offset +
 icb * input_fs_pitch +
 kh * DILATION_SIZE_Y * input_y_pitch +
 xb * input_x_pitch);
 line_cache[xb + 0] = vv[0];
 line_cache[xb + 1] = vv[1];
 line_cache[xb + 2] = vv[2];
 line_cache[xb + 3] = vv[3];
 line_cache[xb + 4] = vv[4];
 line_cache[xb + 5] = vv[5];
 line_cache[xb + 6] = vv[6];
 line_cache[xb + 7] = vv[7];
 }
 for (; xb + 4 <= INPUT_LINE_SIZE; xb += 4) {
 INPUT_TYPE4 vv = DT_INPUT_BLOCK_READ4(input, grouped_input_offset +
 icb * input_fs_pitch +
 kh * DILATION_SIZE_Y * input_y_pitch +
 xb * input_x_pitch);
 line_cache[xb + 0] = vv[0];
 line_cache[xb + 1] = vv[1];
 line_cache[xb + 2] = vv[2];
 line_cache[xb + 3] = vv[3];
 }
 for (; xb < INPUT_LINE_SIZE; xb++) {
 line_cache[xb] = DT_INPUT_BLOCK_READ(input, grouped_input_offset +
 icb * input_fs_pitch +
 kh * DILATION_SIZE_Y * input_y_pitch +
 xb * input_x_pitch);
 }
 }
 __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
 for (int kw = 0; kw < FILTER_SIZE_X; kw++) {
 vec_t src;
 __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_SIZE)))
 for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
#if FILTER_SIZE_X == 1 && DILATION_SIZE_X == 1 && STRIDE_SIZE_X == 1
 src[i] = line_cache[i];
#else
 src[i] = line_cache[kw * DILATION_SIZE_X + STRIDE_SIZE_X * i];
#endif
 }
#if MULTIPLE_GROUPS_INPUT_PRELOAD
 typedef MAKE_VECTOR_TYPE(FILTER_TYPE, FILTER_IFM_NUM) ifm_vec_t;
 ifm_vec_t wei0 = FILTER_VAL_ZERO;
 for (int ifm = 0; ifm < FILTER_IFM_NUM; ifm++)
 wei0[ifm] = weights[grouped_filter_offset +
 ofm_in_group +
 ifm * filter_isv_pitch +
 kh * filter_y_pitch +
 kw * filter_x_pitch];
#if FILTER_IFM_NUM == 2
 const vec_t src0 = GET_SRC(src, g * FILTER_IFM_NUM + 0);
 const vec_t src1 = GET_SRC(src, g * FILTER_IFM_NUM + 1);
 dst = mad(wei0.s0, src0, dst);
 dst = mad(wei0.s1, src1, dst);
#elif FILTER_IFM_NUM == 4
 const vec_t src0 = GET_SRC(src, g * FILTER_IFM_NUM + 0);
 const vec_t src1 = GET_SRC(src, g * FILTER_IFM_NUM + 1);
 const vec_t src2 = GET_SRC(src, g * FILTER_IFM_NUM + 2);
 const vec_t src3 = GET_SRC(src, g * FILTER_IFM_NUM + 3);
 dst = mad(wei0.s0, src0, dst);
 dst = mad(wei0.s1, src1, dst);
 dst = mad(wei0.s2, src2, dst);
 dst = mad(wei0.s3, src3, dst);
#elif FILTER_IFM_NUM == 8
 const vec_t src0 = GET_SRC(src, g * FILTER_IFM_NUM + 0);
 const vec_t src1 = GET_SRC(src, g * FILTER_IFM_NUM + 1);
 const vec_t src2 = GET_SRC(src, g * FILTER_IFM_NUM + 2);
 const vec_t src3 = GET_SRC(src, g * FILTER_IFM_NUM + 3);
 const vec_t src4 = GET_SRC(src, g * FILTER_IFM_NUM + 4);
 const vec_t src5 = GET_SRC(src, g * FILTER_IFM_NUM + 5);
 const vec_t src6 = GET_SRC(src, g * FILTER_IFM_NUM + 6);
 const vec_t src7 = GET_SRC(src, g * FILTER_IFM_NUM + 7);
 dst = mad(wei0.s0, src0, dst);
 dst = mad(wei0.s1, src1, dst);
 dst = mad(wei0.s2, src2, dst);
 dst = mad(wei0.s3, src3, dst);
 dst = mad(wei0.s4, src4, dst);
 dst = mad(wei0.s5, src5, dst);
 dst = mad(wei0.s6, src6, dst);
 dst = mad(wei0.s7, src7, dst);
#else
# error convolution_gpu_bfyx_f16.cl: unsupported input feature size for multiple groups input preload
#endif
#else
 FILTER_TYPE8 wei0 = DT_FILTER_BLOCK_READ8(weights, grouped_filter_offset +
 icb * filter_is_pitch +
 kh * filter_y_pitch +
 kw * filter_x_pitch);
 FILTER_TYPE8 wei1 = DT_FILTER_BLOCK_READ8(weights, grouped_filter_offset +
 icb * filter_is_pitch +
 kh * filter_y_pitch +
 kw * filter_x_pitch +
 8 * filter_isv_pitch);
 const vec_t src0 = GET_SRC(src, 0);
 const vec_t src1 = GET_SRC(src, 1);
 const vec_t src2 = GET_SRC(src, 2);
 const vec_t src3 = GET_SRC(src, 3);
 const vec_t src4 = GET_SRC(src, 4);
 const vec_t src5 = GET_SRC(src, 5);
 const vec_t src6 = GET_SRC(src, 6);
 const vec_t src7 = GET_SRC(src, 7);
 const vec_t src8 = GET_SRC(src, 8);
 const vec_t src9 = GET_SRC(src, 9);
 const vec_t src10 = GET_SRC(src, 10);
 const vec_t src11 = GET_SRC(src, 11);
 const vec_t src12 = GET_SRC(src, 12);
 const vec_t src13 = GET_SRC(src, 13);
 const vec_t src14 = GET_SRC(src, 14);
 const vec_t src15 = GET_SRC(src, 15);
 dst = mad(wei0.s0, src0, dst);
 dst = mad(wei0.s1, src1, dst);
 dst = mad(wei0.s2, src2, dst);
 dst = mad(wei0.s3, src3, dst);
 dst = mad(wei0.s4, src4, dst);
 dst = mad(wei0.s5, src5, dst);
 dst = mad(wei0.s6, src6, dst);
 dst = mad(wei0.s7, src7, dst);
 dst = mad(wei1.s0, src8, dst);
 dst = mad(wei1.s1, src9, dst);
 dst = mad(wei1.s2, src10, dst);
 dst = mad(wei1.s3, src11, dst);
 dst = mad(wei1.s4, src12, dst);
 dst = mad(wei1.s5, src13, dst);
 dst = mad(wei1.s6, src14, dst);
 dst = mad(wei1.s7, src15, dst);
#endif
 }
 }
 }
#if GROUPED && !MULTIPLE_GROUPS_INPUT_PRELOAD
 }
#endif
#if SLM_DIV_FACTOR > 1
 partial_summ[lid1] = dst;
 barrier(CLK_LOCAL_MEM_FENCE);
 if (feature_sub_block == 0) {
 unroll_for(int i = 1; i < SLM_DIV_FACTOR; i++)
 dst += partial_summ[lid1 % feature_per_wg + i * feature_per_wg];
#endif
 dst = ACTIVATION(dst, ACTIVATION_PARAMS);
 typedef MAKE_VECTOR_TYPE(OUTPUT_TYPE, OUTPUT_X_BLOCK_SIZE) out_vec_t;
 out_vec_t res;
#if OUTPUT_LEFTOVERS
 if ((feature_block + 1) * FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM) {
 for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
#if HAS_FUSED_OPS
 FUSED_OPS_SCALAR;
# if OUTPUT_FORMAT_BFYX
 res[i] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SCALAR);
# else
 res[i] = FUSED_OPS_RESULT_SCALAR;
# endif
#else
 res[i] = TO_OUTPUT_TYPE(dst[i]);
#endif
#if OUTPUT_FORMAT_BFYX
 if ((feature_block * FEATURE_SLICE_SIZE + sglid < OUTPUT_FEATURE_NUM) && (x + i) < OUTPUT_SIZE_X) {
 output[output_offset + i] = res[i];
 }
#else
 if ((feature_block * FEATURE_SLICE_SIZE + sglid < OUTPUT_FEATURE_NUM) && (x + i) < OUTPUT_SIZE_X) {
 output[output_offset + i * output_x_pitch + sglid] = res[i];
 }
#endif
 }
 }
 else
#endif
 {
 if (x + OUTPUT_X_BLOCK_SIZE <= OUTPUT_SIZE_X || OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE == 0) {
#if HAS_FUSED_OPS
 FUSED_OPS_VEC;
# if OUTPUT_FORMAT_BFYX
 res = TO_OUTPUTVTYPE(FUSED_OPS_RESULT_VEC);
# else
 res = FUSED_OPS_RESULT_VEC;
# endif
#else
# if OUTPUT_FORMAT_BFYX
 res = TO_OUTPUTVTYPE(dst);
# else
 res = dst;
# endif
#endif
#if OUTPUT_FORMAT_BFYX
 #if OUTPUT_X_BLOCK_SIZE == 2 || OUTPUT_X_BLOCK_SIZE == 4 || OUTPUT_X_BLOCK_SIZE == 8
 VSTORE(res, 0, output + output_offset);
 #elif OUTPUT_X_BLOCK_SIZE == 1
 output[output_offset] = res[0];
 #else
 # error convolution_gpu_bfyx_f16.cl: unsupported output x block size
 #endif
#else
 #if OUTPUT_X_BLOCK_SIZE == 8
 DT_OUTPUT_BLOCK_WRITE8(output, output_offset, res);
 #elif OUTPUT_X_BLOCK_SIZE == 4
 DT_OUTPUT_BLOCK_WRITE4(output, output_offset, res);
 #elif OUTPUT_X_BLOCK_SIZE == 2
 DT_OUTPUT_BLOCK_WRITE2(output, output_offset, res);
 #elif OUTPUT_X_BLOCK_SIZE == 1
 DT_OUTPUT_BLOCK_WRITE(output, output_offset, res);
 #else
 # error convolution_gpu_bfyx_f16.cl: unsupported output x block size
 #endif
#endif
 } else {
 for (int i = 0; i < OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE; i++) {
#if HAS_FUSED_OPS
 FUSED_OPS_SCALAR;
# if OUTPUT_FORMAT_BFYX
 res[i] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SCALAR);
# else
 res[i] = FUSED_OPS_RESULT_SCALAR;
# endif
#else
 res[i] = TO_OUTPUT_TYPE(dst[i]);
#endif
#if OUTPUT_FORMAT_BFYX
 output[output_offset + i] = res[i];
#else
 DT_OUTPUT_BLOCK_WRITE(output, output_offset + i * output_x_pitch, res[i]);
#endif
 }
 }
 }
#if SLM_DIV_FACTOR > 1
 }
#endif
}
#undef AS_INPUT_SRC
#undef AS_US_SRC
#undef GET_SRC
#undef FEATURE_SLICE_SIZE
#undef FILTER_OFM_NUM_ALIGNED
#undef FILTER_IFM_NUM_ALIGNED
#undef INPUT_TYPE
#undef INPUT_TYPE2
#undef INPUT_TYPE4
#undef INPUT_TYPE8
#undef FILTER_TYPE8
#undef AS_INPUT_TYPE
#undef AS_INPUT_TYPE2
#undef AS_INPUT_TYPE4
#undef AS_INPUT_TYPE8
#undef AS_FILTER_TYPE8
#if OUTPUT_FORMAT_BFYX
# undef OUTPUTVTYPE
# undef TO_OUTPUTVTYPE
# undef VSTORE
#endif
#ifdef INPUT_TYPE
#undef INPUT_TYPE
#endif
#ifdef INPUT_TYPE2
#undef INPUT_TYPE2
#endif
#ifdef INPUT_TYPE4
#undef INPUT_TYPE4
#endif
#ifdef INPUT_TYPE8
#undef INPUT_TYPE8
#endif
#ifdef FILTER_TYPE8
#undef FILTER_TYPE8
#endif
#ifdef AS_INPUT_TYPE
#undef AS_INPUT_TYPE
#endif
#ifdef AS_INPUT_TYPE2
#undef AS_INPUT_TYPE2
#endif
#ifdef AS_INPUT_TYPE4
#undef AS_INPUT_TYPE4
#endif
#ifdef AS_INPUT_TYPE8
#undef AS_INPUT_TYPE8
#endif
#ifdef AS_FILTER_TYPE8
#undef AS_FILTER_TYPE8
#endif
#ifdef FEATURE_SLICE_SIZE
#undef FEATURE_SLICE_SIZE
#endif
#ifdef FILTER_OFM_NUM_ALIGNED
#undef FILTER_OFM_NUM_ALIGNED
#endif
#ifdef FILTER_IFM_NUM_ALIGNED
#undef FILTER_IFM_NUM_ALIGNED
#endif
#undef KERNEL
#undef KERNEL_ID
#undef FUNC
#undef FUNC_CALL
#undef CONST_ARRAY_DECL
#undef CONST_ARRAY_REF
#ifdef FP64_SUPPORTED
#undef FP64_SUPPORTED
#endif
#ifdef FP16_SUPPORTED
#undef FP16_SUPPORTED
#endif
#ifdef FP16_UNIT_USED
#undef FP16_UNIT_USED
#endif
#ifdef INT8_UNIT_USED
#undef INT8_UNIT_USED
#endif
#ifdef INT32_UNIT_USED
#undef INT32_UNIT_USED
#endif
#ifdef INT64_UNIT_USED
#undef INT64_UNIT_USED
#endif
#ifdef UINT8_UNIT_USED
#undef UINT8_UNIT_USED
#endif
#ifdef UINT32_UNIT_USED
#undef UINT32_UNIT_USED
#endif
#ifdef UNIT_TYPE
#undef UNIT_TYPE
#endif
#ifdef UNIT_VAL_MAX
#undef UNIT_VAL_MAX
#endif
#ifdef UNIT_VAL_MIN
#undef UNIT_VAL_MIN
#endif
#ifdef UNIT_VAL_ONE
#undef UNIT_VAL_ONE
#endif
#ifdef UNIT_VAL_ZERO
#undef UNIT_VAL_ZERO
#endif
#ifdef TO_UNIT_TYPE
#undef TO_UNIT_TYPE
#endif
#ifdef TO_UNIT_TYPE_SAT
#undef TO_UNIT_TYPE_SAT
#endif
#ifdef AS_UNIT_TYPE
#undef AS_UNIT_TYPE
#endif
#ifdef UNIT_MAX_FUNC
#undef UNIT_MAX_FUNC
#endif
#ifdef UNIT_MIN_FUNC
#undef UNIT_MIN_FUNC
#endif
#ifdef UNIT_ABS_FUNC
#undef UNIT_ABS_FUNC
#endif
#ifdef UNIT_TYPE_SIZE
#undef UNIT_TYPE_SIZE
#endif
#ifdef UNIT_IS_FP
#undef UNIT_IS_FP
#endif
#ifdef NL_M
#undef NL_M
#endif
#ifdef NL_N
#undef NL_N
#endif
#ifdef ACTIVATION_FUNC_TYPE
#undef ACTIVATION_FUNC_TYPE
#endif
#ifdef ACTIVATION_FUNC_VAL_MAX
#undef ACTIVATION_FUNC_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_VAL_MIN
#undef ACTIVATION_FUNC_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_VAL_ONE
#undef ACTIVATION_FUNC_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_VAL_ZERO
#undef ACTIVATION_FUNC_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_TYPE
#undef TO_ACTIVATION_FUNC_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_TYPE_SAT
#undef TO_ACTIVATION_FUNC_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_TYPE
#undef AS_ACTIVATION_FUNC_TYPE
#endif
#ifdef ACTIVATION_FUNC_MAX_FUNC
#undef ACTIVATION_FUNC_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_MIN_FUNC
#undef ACTIVATION_FUNC_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_ABS_FUNC
#undef ACTIVATION_FUNC_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_TYPE_SIZE
#undef ACTIVATION_FUNC_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_IS_FP
#undef ACTIVATION_FUNC_IS_FP
#endif
#ifdef ACTIVATION_PARAMS
#undef ACTIVATION_PARAMS
#endif
#ifdef ACTIVATION_FUNC
#undef ACTIVATION_FUNC
#endif
#ifdef ACTIVATION
#undef ACTIVATION
#endif
#ifdef INPUT0_SIZE_X
#undef INPUT0_SIZE_X
#endif
#ifdef INPUT0_SIZE_Y
#undef INPUT0_SIZE_Y
#endif
#ifdef INPUT0_SIZE_Z
#undef INPUT0_SIZE_Z
#endif
#ifdef INPUT0_SIZE_W
#undef INPUT0_SIZE_W
#endif
#ifdef INPUT0_SIZE_U
#undef INPUT0_SIZE_U
#endif
#ifdef INPUT0_SIZE_V
#undef INPUT0_SIZE_V
#endif
#ifdef INPUT0_FEATURE_NUM
#undef INPUT0_FEATURE_NUM
#endif
#ifdef INPUT0_BATCH_NUM
#undef INPUT0_BATCH_NUM
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_X
#undef INPUT0_PAD_BEFORE_SIZE_X
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_Y
#undef INPUT0_PAD_BEFORE_SIZE_Y
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_Z
#undef INPUT0_PAD_BEFORE_SIZE_Z
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_W
#undef INPUT0_PAD_BEFORE_SIZE_W
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_U
#undef INPUT0_PAD_BEFORE_SIZE_U
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_V
#undef INPUT0_PAD_BEFORE_SIZE_V
#endif
#ifdef INPUT0_PAD_BEFORE_FEATURE_NUM
#undef INPUT0_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef INPUT0_PAD_BEFORE_BATCH_NUM
#undef INPUT0_PAD_BEFORE_BATCH_NUM
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_X
#undef INPUT0_PAD_AFTER_SIZE_X
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_Y
#undef INPUT0_PAD_AFTER_SIZE_Y
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_Z
#undef INPUT0_PAD_AFTER_SIZE_Z
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_W
#undef INPUT0_PAD_AFTER_SIZE_W
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_U
#undef INPUT0_PAD_AFTER_SIZE_U
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_V
#undef INPUT0_PAD_AFTER_SIZE_V
#endif
#ifdef INPUT0_PAD_AFTER_FEATURE_NUM
#undef INPUT0_PAD_AFTER_FEATURE_NUM
#endif
#ifdef INPUT0_PAD_AFTER_BATCH_NUM
#undef INPUT0_PAD_AFTER_BATCH_NUM
#endif
#ifdef INPUT0_LENGTH
#undef INPUT0_LENGTH
#endif
#ifdef INPUT0_X_PITCH
#undef INPUT0_X_PITCH
#endif
#ifdef INPUT0_Y_PITCH
#undef INPUT0_Y_PITCH
#endif
#ifdef INPUT0_Z_PITCH
#undef INPUT0_Z_PITCH
#endif
#ifdef INPUT0_W_PITCH
#undef INPUT0_W_PITCH
#endif
#ifdef INPUT0_U_PITCH
#undef INPUT0_U_PITCH
#endif
#ifdef INPUT0_V_PITCH
#undef INPUT0_V_PITCH
#endif
#ifdef INPUT0_FEATURE_PITCH
#undef INPUT0_FEATURE_PITCH
#endif
#ifdef INPUT0_BATCH_PITCH
#undef INPUT0_BATCH_PITCH
#endif
#ifdef INPUT0_GET_INDEX_SAFE
#undef INPUT0_GET_INDEX_SAFE
#endif
#ifdef INPUT0_GET_INDEX
#undef INPUT0_GET_INDEX
#endif
#ifdef INPUT0_GET_INDEX_RAW
#undef INPUT0_GET_INDEX_RAW
#endif
#ifdef INPUT0_VIEW_OFFSET
#undef INPUT0_VIEW_OFFSET
#endif
#ifdef INPUT0_DIMS
#undef INPUT0_DIMS
#endif
#ifdef INPUT0_SIMPLE
#undef INPUT0_SIMPLE
#endif
#ifdef INPUT0_GROUPED
#undef INPUT0_GROUPED
#endif
#ifdef INPUT0_LAYOUT_B_FS_YX_FSV16
#undef INPUT0_LAYOUT_B_FS_YX_FSV16
#endif
#ifdef INPUT0_TYPE
#undef INPUT0_TYPE
#endif
#ifdef INPUT0_VAL_MAX
#undef INPUT0_VAL_MAX
#endif
#ifdef INPUT0_VAL_MIN
#undef INPUT0_VAL_MIN
#endif
#ifdef INPUT0_VAL_ONE
#undef INPUT0_VAL_ONE
#endif
#ifdef INPUT0_VAL_ZERO
#undef INPUT0_VAL_ZERO
#endif
#ifdef TO_INPUT0_TYPE
#undef TO_INPUT0_TYPE
#endif
#ifdef TO_INPUT0_TYPE_SAT
#undef TO_INPUT0_TYPE_SAT
#endif
#ifdef AS_INPUT0_TYPE
#undef AS_INPUT0_TYPE
#endif
#ifdef INPUT0_MAX_FUNC
#undef INPUT0_MAX_FUNC
#endif
#ifdef INPUT0_MIN_FUNC
#undef INPUT0_MIN_FUNC
#endif
#ifdef INPUT0_ABS_FUNC
#undef INPUT0_ABS_FUNC
#endif
#ifdef INPUT0_TYPE_SIZE
#undef INPUT0_TYPE_SIZE
#endif
#ifdef INPUT0_IS_FP
#undef INPUT0_IS_FP
#endif
#ifdef INPUT0_OFFSET
#undef INPUT0_OFFSET
#endif
#ifdef INPUT0_SIZES_DATA
#undef INPUT0_SIZES_DATA
#endif
#ifdef INPUT0_PITCHES
#undef INPUT0_PITCHES
#endif
#ifdef INPUT0_PAD_BEFORE
#undef INPUT0_PAD_BEFORE
#endif
#ifdef INPUT0_PAD_AFTER
#undef INPUT0_PAD_AFTER
#endif
#ifdef OUTPUT_SIZE_X
#undef OUTPUT_SIZE_X
#endif
#ifdef OUTPUT_SIZE_Y
#undef OUTPUT_SIZE_Y
#endif
#ifdef OUTPUT_SIZE_Z
#undef OUTPUT_SIZE_Z
#endif
#ifdef OUTPUT_SIZE_W
#undef OUTPUT_SIZE_W
#endif
#ifdef OUTPUT_SIZE_U
#undef OUTPUT_SIZE_U
#endif
#ifdef OUTPUT_SIZE_V
#undef OUTPUT_SIZE_V
#endif
#ifdef OUTPUT_FEATURE_NUM
#undef OUTPUT_FEATURE_NUM
#endif
#ifdef OUTPUT_BATCH_NUM
#undef OUTPUT_BATCH_NUM
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_X
#undef OUTPUT_PAD_BEFORE_SIZE_X
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_Y
#undef OUTPUT_PAD_BEFORE_SIZE_Y
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_Z
#undef OUTPUT_PAD_BEFORE_SIZE_Z
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_W
#undef OUTPUT_PAD_BEFORE_SIZE_W
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_U
#undef OUTPUT_PAD_BEFORE_SIZE_U
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_V
#undef OUTPUT_PAD_BEFORE_SIZE_V
#endif
#ifdef OUTPUT_PAD_BEFORE_FEATURE_NUM
#undef OUTPUT_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef OUTPUT_PAD_BEFORE_BATCH_NUM
#undef OUTPUT_PAD_BEFORE_BATCH_NUM
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_X
#undef OUTPUT_PAD_AFTER_SIZE_X
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_Y
#undef OUTPUT_PAD_AFTER_SIZE_Y
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_Z
#undef OUTPUT_PAD_AFTER_SIZE_Z
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_W
#undef OUTPUT_PAD_AFTER_SIZE_W
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_U
#undef OUTPUT_PAD_AFTER_SIZE_U
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_V
#undef OUTPUT_PAD_AFTER_SIZE_V
#endif
#ifdef OUTPUT_PAD_AFTER_FEATURE_NUM
#undef OUTPUT_PAD_AFTER_FEATURE_NUM
#endif
#ifdef OUTPUT_PAD_AFTER_BATCH_NUM
#undef OUTPUT_PAD_AFTER_BATCH_NUM
#endif
#ifdef OUTPUT_LENGTH
#undef OUTPUT_LENGTH
#endif
#ifdef OUTPUT_X_PITCH
#undef OUTPUT_X_PITCH
#endif
#ifdef OUTPUT_Y_PITCH
#undef OUTPUT_Y_PITCH
#endif
#ifdef OUTPUT_Z_PITCH
#undef OUTPUT_Z_PITCH
#endif
#ifdef OUTPUT_W_PITCH
#undef OUTPUT_W_PITCH
#endif
#ifdef OUTPUT_U_PITCH
#undef OUTPUT_U_PITCH
#endif
#ifdef OUTPUT_V_PITCH
#undef OUTPUT_V_PITCH
#endif
#ifdef OUTPUT_FEATURE_PITCH
#undef OUTPUT_FEATURE_PITCH
#endif
#ifdef OUTPUT_BATCH_PITCH
#undef OUTPUT_BATCH_PITCH
#endif
#ifdef OUTPUT_GET_INDEX_SAFE
#undef OUTPUT_GET_INDEX_SAFE
#endif
#ifdef OUTPUT_GET_INDEX
#undef OUTPUT_GET_INDEX
#endif
#ifdef OUTPUT_GET_INDEX_RAW
#undef OUTPUT_GET_INDEX_RAW
#endif
#ifdef OUTPUT_VIEW_OFFSET
#undef OUTPUT_VIEW_OFFSET
#endif
#ifdef OUTPUT_DIMS
#undef OUTPUT_DIMS
#endif
#ifdef OUTPUT_SIMPLE
#undef OUTPUT_SIMPLE
#endif
#ifdef OUTPUT_GROUPED
#undef OUTPUT_GROUPED
#endif
#ifdef OUTPUT_LAYOUT_B_FS_YX_FSV16
#undef OUTPUT_LAYOUT_B_FS_YX_FSV16
#endif
#ifdef OUTPUT_TYPE
#undef OUTPUT_TYPE
#endif
#ifdef OUTPUT_VAL_MAX
#undef OUTPUT_VAL_MAX
#endif
#ifdef OUTPUT_VAL_MIN
#undef OUTPUT_VAL_MIN
#endif
#ifdef OUTPUT_VAL_ONE
#undef OUTPUT_VAL_ONE
#endif
#ifdef OUTPUT_VAL_ZERO
#undef OUTPUT_VAL_ZERO
#endif
#ifdef TO_OUTPUT_TYPE
#undef TO_OUTPUT_TYPE
#endif
#ifdef TO_OUTPUT_TYPE_SAT
#undef TO_OUTPUT_TYPE_SAT
#endif
#ifdef AS_OUTPUT_TYPE
#undef AS_OUTPUT_TYPE
#endif
#ifdef OUTPUT_MAX_FUNC
#undef OUTPUT_MAX_FUNC
#endif
#ifdef OUTPUT_MIN_FUNC
#undef OUTPUT_MIN_FUNC
#endif
#ifdef OUTPUT_ABS_FUNC
#undef OUTPUT_ABS_FUNC
#endif
#ifdef OUTPUT_TYPE_SIZE
#undef OUTPUT_TYPE_SIZE
#endif
#ifdef OUTPUT_IS_FP
#undef OUTPUT_IS_FP
#endif
#ifdef OUTPUT_OFFSET
#undef OUTPUT_OFFSET
#endif
#ifdef OUTPUT_SIZES_DATA
#undef OUTPUT_SIZES_DATA
#endif
#ifdef OUTPUT_PITCHES
#undef OUTPUT_PITCHES
#endif
#ifdef OUTPUT_PAD_BEFORE
#undef OUTPUT_PAD_BEFORE
#endif
#ifdef OUTPUT_PAD_AFTER
#undef OUTPUT_PAD_AFTER
#endif
#ifdef OPTIONAL_SHAPE_INFO_ARG
#undef OPTIONAL_SHAPE_INFO_ARG
#endif
#ifdef OPTIONAL_SHAPE_INFO_TENSOR
#undef OPTIONAL_SHAPE_INFO_TENSOR
#endif
#ifdef FILTER_SIZE_X
#undef FILTER_SIZE_X
#endif
#ifdef FILTER_SIZE_Y
#undef FILTER_SIZE_Y
#endif
#ifdef FILTER_SIZE_Z
#undef FILTER_SIZE_Z
#endif
#ifdef FILTER_IFM_NUM
#undef FILTER_IFM_NUM
#endif
#ifdef FILTER_OFM_NUM
#undef FILTER_OFM_NUM
#endif
#ifdef FILTER_GROUPS_NUM
#undef FILTER_GROUPS_NUM
#endif
#ifdef FILTER_X_PITCH
#undef FILTER_X_PITCH
#endif
#ifdef FILTER_Y_PITCH
#undef FILTER_Y_PITCH
#endif
#ifdef FILTER_Z_PITCH
#undef FILTER_Z_PITCH
#endif
#ifdef FILTER_IFM_PITCH
#undef FILTER_IFM_PITCH
#endif
#ifdef FILTER_OFM_PITCH
#undef FILTER_OFM_PITCH
#endif
#ifdef FILTER_GROUPS_PITCH
#undef FILTER_GROUPS_PITCH
#endif
#ifdef FILTER_LENGTH
#undef FILTER_LENGTH
#endif
#ifdef FILTER_VIEW_OFFSET
#undef FILTER_VIEW_OFFSET
#endif
#ifdef FILTER_DIMS
#undef FILTER_DIMS
#endif
#ifdef FILTER_SIMPLE
#undef FILTER_SIMPLE
#endif
#ifdef FILTER_GROUPED
#undef FILTER_GROUPED
#endif
#ifdef FILTER_LAYOUT_OS_IS_YX_ISV16_OSV16
#undef FILTER_LAYOUT_OS_IS_YX_ISV16_OSV16
#endif
#ifdef FILTER_TYPE
#undef FILTER_TYPE
#endif
#ifdef FILTER_VAL_MAX
#undef FILTER_VAL_MAX
#endif
#ifdef FILTER_VAL_MIN
#undef FILTER_VAL_MIN
#endif
#ifdef FILTER_VAL_ONE
#undef FILTER_VAL_ONE
#endif
#ifdef FILTER_VAL_ZERO
#undef FILTER_VAL_ZERO
#endif
#ifdef TO_FILTER_TYPE
#undef TO_FILTER_TYPE
#endif
#ifdef TO_FILTER_TYPE_SAT
#undef TO_FILTER_TYPE_SAT
#endif
#ifdef AS_FILTER_TYPE
#undef AS_FILTER_TYPE
#endif
#ifdef FILTER_MAX_FUNC
#undef FILTER_MAX_FUNC
#endif
#ifdef FILTER_MIN_FUNC
#undef FILTER_MIN_FUNC
#endif
#ifdef FILTER_ABS_FUNC
#undef FILTER_ABS_FUNC
#endif
#ifdef FILTER_TYPE_SIZE
#undef FILTER_TYPE_SIZE
#endif
#ifdef FILTER_IS_FP
#undef FILTER_IS_FP
#endif
#ifdef FILTER_OFFSET
#undef FILTER_OFFSET
#endif
#ifdef FILTER_SIZES_DATA
#undef FILTER_SIZES_DATA
#endif
#ifdef FILTER_PITCHES
#undef FILTER_PITCHES
#endif
#ifdef FILTER_PAD_BEFORE
#undef FILTER_PAD_BEFORE
#endif
#ifdef FILTER_PAD_AFTER
#undef FILTER_PAD_AFTER
#endif
#ifdef BIAS_TERM
#undef BIAS_TERM
#endif
#ifdef BIAS_SIZE_X
#undef BIAS_SIZE_X
#endif
#ifdef BIAS_SIZE_Y
#undef BIAS_SIZE_Y
#endif
#ifdef BIAS_SIZE_Z
#undef BIAS_SIZE_Z
#endif
#ifdef BIAS_SIZE_W
#undef BIAS_SIZE_W
#endif
#ifdef BIAS_SIZE_U
#undef BIAS_SIZE_U
#endif
#ifdef BIAS_SIZE_V
#undef BIAS_SIZE_V
#endif
#ifdef BIAS_FEATURE_NUM
#undef BIAS_FEATURE_NUM
#endif
#ifdef BIAS_BATCH_NUM
#undef BIAS_BATCH_NUM
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_X
#undef BIAS_PAD_BEFORE_SIZE_X
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_Y
#undef BIAS_PAD_BEFORE_SIZE_Y
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_Z
#undef BIAS_PAD_BEFORE_SIZE_Z
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_W
#undef BIAS_PAD_BEFORE_SIZE_W
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_U
#undef BIAS_PAD_BEFORE_SIZE_U
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_V
#undef BIAS_PAD_BEFORE_SIZE_V
#endif
#ifdef BIAS_PAD_BEFORE_FEATURE_NUM
#undef BIAS_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef BIAS_PAD_BEFORE_BATCH_NUM
#undef BIAS_PAD_BEFORE_BATCH_NUM
#endif
#ifdef BIAS_PAD_AFTER_SIZE_X
#undef BIAS_PAD_AFTER_SIZE_X
#endif
#ifdef BIAS_PAD_AFTER_SIZE_Y
#undef BIAS_PAD_AFTER_SIZE_Y
#endif
#ifdef BIAS_PAD_AFTER_SIZE_Z
#undef BIAS_PAD_AFTER_SIZE_Z
#endif
#ifdef BIAS_PAD_AFTER_SIZE_W
#undef BIAS_PAD_AFTER_SIZE_W
#endif
#ifdef BIAS_PAD_AFTER_SIZE_U
#undef BIAS_PAD_AFTER_SIZE_U
#endif
#ifdef BIAS_PAD_AFTER_SIZE_V
#undef BIAS_PAD_AFTER_SIZE_V
#endif
#ifdef BIAS_PAD_AFTER_FEATURE_NUM
#undef BIAS_PAD_AFTER_FEATURE_NUM
#endif
#ifdef BIAS_PAD_AFTER_BATCH_NUM
#undef BIAS_PAD_AFTER_BATCH_NUM
#endif
#ifdef BIAS_LENGTH
#undef BIAS_LENGTH
#endif
#ifdef BIAS_X_PITCH
#undef BIAS_X_PITCH
#endif
#ifdef BIAS_Y_PITCH
#undef BIAS_Y_PITCH
#endif
#ifdef BIAS_Z_PITCH
#undef BIAS_Z_PITCH
#endif
#ifdef BIAS_W_PITCH
#undef BIAS_W_PITCH
#endif
#ifdef BIAS_U_PITCH
#undef BIAS_U_PITCH
#endif
#ifdef BIAS_V_PITCH
#undef BIAS_V_PITCH
#endif
#ifdef BIAS_FEATURE_PITCH
#undef BIAS_FEATURE_PITCH
#endif
#ifdef BIAS_BATCH_PITCH
#undef BIAS_BATCH_PITCH
#endif
#ifdef BIAS_GET_INDEX_SAFE
#undef BIAS_GET_INDEX_SAFE
#endif
#ifdef BIAS_GET_INDEX
#undef BIAS_GET_INDEX
#endif
#ifdef BIAS_GET_INDEX_RAW
#undef BIAS_GET_INDEX_RAW
#endif
#ifdef BIAS_VIEW_OFFSET
#undef BIAS_VIEW_OFFSET
#endif
#ifdef BIAS_DIMS
#undef BIAS_DIMS
#endif
#ifdef BIAS_SIMPLE
#undef BIAS_SIMPLE
#endif
#ifdef BIAS_GROUPED
#undef BIAS_GROUPED
#endif
#ifdef BIAS_LAYOUT_BF
#undef BIAS_LAYOUT_BF
#endif
#ifdef BIAS_TYPE
#undef BIAS_TYPE
#endif
#ifdef BIAS_VAL_MAX
#undef BIAS_VAL_MAX
#endif
#ifdef BIAS_VAL_MIN
#undef BIAS_VAL_MIN
#endif
#ifdef BIAS_VAL_ONE
#undef BIAS_VAL_ONE
#endif
#ifdef BIAS_VAL_ZERO
#undef BIAS_VAL_ZERO
#endif
#ifdef TO_BIAS_TYPE
#undef TO_BIAS_TYPE
#endif
#ifdef TO_BIAS_TYPE_SAT
#undef TO_BIAS_TYPE_SAT
#endif
#ifdef AS_BIAS_TYPE
#undef AS_BIAS_TYPE
#endif
#ifdef BIAS_MAX_FUNC
#undef BIAS_MAX_FUNC
#endif
#ifdef BIAS_MIN_FUNC
#undef BIAS_MIN_FUNC
#endif
#ifdef BIAS_ABS_FUNC
#undef BIAS_ABS_FUNC
#endif
#ifdef BIAS_TYPE_SIZE
#undef BIAS_TYPE_SIZE
#endif
#ifdef BIAS_IS_FP
#undef BIAS_IS_FP
#endif
#ifdef BIAS_OFFSET
#undef BIAS_OFFSET
#endif
#ifdef BIAS_SIZES_DATA
#undef BIAS_SIZES_DATA
#endif
#ifdef BIAS_PITCHES
#undef BIAS_PITCHES
#endif
#ifdef BIAS_PAD_BEFORE
#undef BIAS_PAD_BEFORE
#endif
#ifdef BIAS_PAD_AFTER
#undef BIAS_PAD_AFTER
#endif
#ifdef BIAS_PER_OUTPUT
#undef BIAS_PER_OUTPUT
#endif
#ifdef BIAS_PER_OFM
#undef BIAS_PER_OFM
#endif
#ifdef STRIDE_SIZE_X
#undef STRIDE_SIZE_X
#endif
#ifdef STRIDE_SIZE_Y
#undef STRIDE_SIZE_Y
#endif
#ifdef STRIDE_SIZE_Z
#undef STRIDE_SIZE_Z
#endif
#ifdef PADDING_SIZE_X
#undef PADDING_SIZE_X
#endif
#ifdef PADDING_SIZE_Y
#undef PADDING_SIZE_Y
#endif
#ifdef PADDING_SIZE_Z
#undef PADDING_SIZE_Z
#endif
#ifdef DILATION_SIZE_X
#undef DILATION_SIZE_X
#endif
#ifdef DILATION_SIZE_Y
#undef DILATION_SIZE_Y
#endif
#ifdef DILATION_SIZE_Z
#undef DILATION_SIZE_Z
#endif
#ifdef FILTER_ARRAY_NUM
#undef FILTER_ARRAY_NUM
#endif
#ifdef INPUT0_OFFSET_WITH_PADDING
#undef INPUT0_OFFSET_WITH_PADDING
#endif
#ifdef GROUPED
#undef GROUPED
#endif
#ifdef SKIP_BATCH
#undef SKIP_BATCH
#endif
#ifdef FUSED_OP0_LOAD_VEC
#undef FUSED_OP0_LOAD_VEC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_MAX
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_MIN
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_ONE
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_ZERO
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE
#undef TO_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE_SAT
#undef TO_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE
#undef AS_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_MAX_FUNC
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_MIN_FUNC
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_ABS_FUNC
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE_SIZE
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_IS_FP
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_IS_FP
#endif
#ifdef ACTIVATION_PARAMS_FUSED_OP0_VEC
#undef ACTIVATION_PARAMS_FUSED_OP0_VEC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC
#undef ACTIVATION_FUNC_FUSED_OP0_VEC
#endif
#ifdef ACTIVATION_FUSED_OP0_VEC
#undef ACTIVATION_FUSED_OP0_VEC
#endif
#ifdef FUSED_OP0_ACTION_VEC
#undef FUSED_OP0_ACTION_VEC
#endif
#ifdef FUSED_OPS_VEC
#undef FUSED_OPS_VEC
#endif
#ifdef FUSED_OPS_PRELOAD_VEC
#undef FUSED_OPS_PRELOAD_VEC
#endif
#ifdef FUSED_OPS_CALC_VEC
#undef FUSED_OPS_CALC_VEC
#endif
#ifdef FUSED_OPS_RESULT_VEC
#undef FUSED_OPS_RESULT_VEC
#endif
#ifdef FUSED_OPS_CAN_USE_PRELOAD_VEC
#undef FUSED_OPS_CAN_USE_PRELOAD_VEC
#endif
#ifdef FUSED_OP0_LOAD_SCALAR
#undef FUSED_OP0_LOAD_SCALAR
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_MAX
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_MIN
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_ONE
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_ZERO
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE
#undef TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE_SAT
#undef TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE
#undef AS_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_MAX_FUNC
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_MIN_FUNC
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_ABS_FUNC
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE_SIZE
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_IS_FP
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_IS_FP
#endif
#ifdef ACTIVATION_PARAMS_FUSED_OP0_SCALAR
#undef ACTIVATION_PARAMS_FUSED_OP0_SCALAR
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR
#endif
#ifdef ACTIVATION_FUSED_OP0_SCALAR
#undef ACTIVATION_FUSED_OP0_SCALAR
#endif
#ifdef FUSED_OP0_ACTION_SCALAR
#undef FUSED_OP0_ACTION_SCALAR
#endif
#ifdef FUSED_OPS_SCALAR
#undef FUSED_OPS_SCALAR
#endif
#ifdef FUSED_OPS_PRELOAD_SCALAR
#undef FUSED_OPS_PRELOAD_SCALAR
#endif
#ifdef FUSED_OPS_CALC_SCALAR
#undef FUSED_OPS_CALC_SCALAR
#endif
#ifdef FUSED_OPS_RESULT_SCALAR
#undef FUSED_OPS_RESULT_SCALAR
#endif
#ifdef FUSED_OPS_CAN_USE_PRELOAD_SCALAR
#undef FUSED_OPS_CAN_USE_PRELOAD_SCALAR
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_X
#undef FUSED_OP_0_OUTPUT_SIZE_X
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_Y
#undef FUSED_OP_0_OUTPUT_SIZE_Y
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_Z
#undef FUSED_OP_0_OUTPUT_SIZE_Z
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_W
#undef FUSED_OP_0_OUTPUT_SIZE_W
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_U
#undef FUSED_OP_0_OUTPUT_SIZE_U
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_V
#undef FUSED_OP_0_OUTPUT_SIZE_V
#endif
#ifdef FUSED_OP_0_OUTPUT_FEATURE_NUM
#undef FUSED_OP_0_OUTPUT_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_BATCH_NUM
#undef FUSED_OP_0_OUTPUT_BATCH_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_X
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_X
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Y
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Y
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Z
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Z
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_W
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_W
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_U
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_U
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_V
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_V
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_FEATURE_NUM
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_BATCH_NUM
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_BATCH_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_X
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_X
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Y
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Y
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Z
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Z
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_W
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_W
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_U
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_U
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_V
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_V
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_FEATURE_NUM
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_BATCH_NUM
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_BATCH_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_LENGTH
#undef FUSED_OP_0_OUTPUT_LENGTH
#endif
#ifdef FUSED_OP_0_OUTPUT_X_PITCH
#undef FUSED_OP_0_OUTPUT_X_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_Y_PITCH
#undef FUSED_OP_0_OUTPUT_Y_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_Z_PITCH
#undef FUSED_OP_0_OUTPUT_Z_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_W_PITCH
#undef FUSED_OP_0_OUTPUT_W_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_U_PITCH
#undef FUSED_OP_0_OUTPUT_U_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_V_PITCH
#undef FUSED_OP_0_OUTPUT_V_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_FEATURE_PITCH
#undef FUSED_OP_0_OUTPUT_FEATURE_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_BATCH_PITCH
#undef FUSED_OP_0_OUTPUT_BATCH_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_GET_INDEX_SAFE
#undef FUSED_OP_0_OUTPUT_GET_INDEX_SAFE
#endif
#ifdef FUSED_OP_0_OUTPUT_GET_INDEX
#undef FUSED_OP_0_OUTPUT_GET_INDEX
#endif
#ifdef FUSED_OP_0_OUTPUT_GET_INDEX_RAW
#undef FUSED_OP_0_OUTPUT_GET_INDEX_RAW
#endif
#ifdef FUSED_OP_0_OUTPUT_VIEW_OFFSET
#undef FUSED_OP_0_OUTPUT_VIEW_OFFSET
#endif
#ifdef FUSED_OP_0_OUTPUT_DIMS
#undef FUSED_OP_0_OUTPUT_DIMS
#endif
#ifdef FUSED_OP_0_OUTPUT_SIMPLE
#undef FUSED_OP_0_OUTPUT_SIMPLE
#endif
#ifdef FUSED_OP_0_OUTPUT_GROUPED
#undef FUSED_OP_0_OUTPUT_GROUPED
#endif
#ifdef FUSED_OP_0_OUTPUT_LAYOUT_BFYX
#undef FUSED_OP_0_OUTPUT_LAYOUT_BFYX
#endif
#ifdef FUSED_OP_0_OUTPUT_TYPE
#undef FUSED_OP_0_OUTPUT_TYPE
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_MAX
#undef FUSED_OP_0_OUTPUT_VAL_MAX
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_MIN
#undef FUSED_OP_0_OUTPUT_VAL_MIN
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_ONE
#undef FUSED_OP_0_OUTPUT_VAL_ONE
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_ZERO
#undef FUSED_OP_0_OUTPUT_VAL_ZERO
#endif
#ifdef TO_FUSED_OP_0_OUTPUT_TYPE
#undef TO_FUSED_OP_0_OUTPUT_TYPE
#endif
#ifdef TO_FUSED_OP_0_OUTPUT_TYPE_SAT
#undef TO_FUSED_OP_0_OUTPUT_TYPE_SAT
#endif
#ifdef AS_FUSED_OP_0_OUTPUT_TYPE
#undef AS_FUSED_OP_0_OUTPUT_TYPE
#endif
#ifdef FUSED_OP_0_OUTPUT_MAX_FUNC
#undef FUSED_OP_0_OUTPUT_MAX_FUNC
#endif
#ifdef FUSED_OP_0_OUTPUT_MIN_FUNC
#undef FUSED_OP_0_OUTPUT_MIN_FUNC
#endif
#ifdef FUSED_OP_0_OUTPUT_ABS_FUNC
#undef FUSED_OP_0_OUTPUT_ABS_FUNC
#endif
#ifdef FUSED_OP_0_OUTPUT_TYPE_SIZE
#undef FUSED_OP_0_OUTPUT_TYPE_SIZE
#endif
#ifdef FUSED_OP_0_OUTPUT_IS_FP
#undef FUSED_OP_0_OUTPUT_IS_FP
#endif
#ifdef FUSED_OP_0_OUTPUT_OFFSET
#undef FUSED_OP_0_OUTPUT_OFFSET
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZES_DATA
#undef FUSED_OP_0_OUTPUT_SIZES_DATA
#endif
#ifdef FUSED_OP_0_OUTPUT_PITCHES
#undef FUSED_OP_0_OUTPUT_PITCHES
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER
#undef FUSED_OP_0_OUTPUT_PAD_AFTER
#endif
#ifdef FUSED_OP0_DECLS
#undef FUSED_OP0_DECLS
#endif
#ifdef FUSED_OP0_ARGS
#undef FUSED_OP0_ARGS
#endif
#ifdef FUSED_OPS_DECLS
#undef FUSED_OPS_DECLS
#endif
#ifdef FUSED_OPS_ARGS
#undef FUSED_OPS_ARGS
#endif
#ifdef HAS_FUSED_OPS
#undef HAS_FUSED_OPS
#endif
#ifdef HAS_FUSED_OPS_DECLS
#undef HAS_FUSED_OPS_DECLS
#endif
#ifdef OUTPUT_X_BLOCK_SIZE
#undef OUTPUT_X_BLOCK_SIZE
#endif
#ifdef INPUT_LINE_SIZE
#undef INPUT_LINE_SIZE
#endif
#ifdef SUB_GROUP_SIZE
#undef SUB_GROUP_SIZE
#endif
#ifdef X_BLOCKS
#undef X_BLOCKS
#endif
#ifdef SLM_DIV_FACTOR
#undef SLM_DIV_FACTOR
#endif
#ifdef WORK_GROUP_SIZE
#undef WORK_GROUP_SIZE
#endif
#ifdef IC_BLOCKS
#undef IC_BLOCKS
#endif

//====================================================
// Kernel template: reorder_data_fast_b1 
// Kernel name: reorder_data_fast_b1_12550024482704781809_0_0
#define KERNEL(name) __kernel void reorder_data_fast_b1_12550024482704781809_0_0
#define KERNEL_ID reorder_data_fast_b1_12550024482704781809_0_0
#define FUNC(name)  _##name##_reorder_data_fast_b1_12550024482704781809_0_0
#define FUNC_CALL(name)  _##name##_reorder_data_fast_b1_12550024482704781809_0_0
#define CONST_ARRAY_DECL(name) __constant size_t  _##name##_reorder_data_fast_b1_12550024482704781809_0_0 []
#define CONST_ARRAY_REF(name)  _##name##_reorder_data_fast_b1_12550024482704781809_0_0
#define FP64_SUPPORTED 0
#define FP16_SUPPORTED 1
#define FP16_UNIT_USED 1
#define INT8_UNIT_USED 0
#define INT32_UNIT_USED 0
#define INT64_UNIT_USED 0
#define UINT8_UNIT_USED 0
#define UINT32_UNIT_USED 0
#define UNIT_TYPE half
#define UNIT_VAL_MAX HALF_MAX
#define UNIT_VAL_MIN -UNIT_VAL_MAX
#define UNIT_VAL_ONE 1.0h
#define UNIT_VAL_ZERO 0.0h
#define TO_UNIT_TYPE(v) convert_half(v)
#define TO_UNIT_TYPE_SAT(v) convert_half(v)
#define AS_UNIT_TYPE(v) as_half(v)
#define UNIT_MAX_FUNC fmax
#define UNIT_MIN_FUNC fmin
#define UNIT_ABS_FUNC fabs
#define UNIT_TYPE_SIZE 2
#define UNIT_IS_FP 1
#define NL_M as_float(0x0)/*0.000000e+00*/
#define NL_N as_float(0x0)/*0.000000e+00*/
#define ACTIVATION_FUNC_TYPE float
#define ACTIVATION_FUNC_VAL_MAX FLT_MAX
#define ACTIVATION_FUNC_VAL_MIN -ACTIVATION_FUNC_VAL_MAX
#define ACTIVATION_FUNC_VAL_ONE 1.0f
#define ACTIVATION_FUNC_VAL_ZERO 0.0f
#define TO_ACTIVATION_FUNC_TYPE(v) convert_float(v)
#define TO_ACTIVATION_FUNC_TYPE_SAT(v) convert_float(v)
#define AS_ACTIVATION_FUNC_TYPE(v) as_float(v)
#define ACTIVATION_FUNC_MAX_FUNC fmax
#define ACTIVATION_FUNC_MIN_FUNC fmin
#define ACTIVATION_FUNC_ABS_FUNC fabs
#define ACTIVATION_FUNC_TYPE_SIZE 4
#define ACTIVATION_FUNC_IS_FP 1
#define ACTIVATION_PARAMS NL_M, NL_N
#define ACTIVATION_FUNC(input, m, n) input
#define ACTIVATION(input, params) ACTIVATION_FUNC(input, params)
#define INPUT0_SIZE_X 1
#define INPUT0_SIZE_Y 1
#define INPUT0_SIZE_Z 1
#define INPUT0_SIZE_W 1
#define INPUT0_SIZE_U 1
#define INPUT0_SIZE_V 1
#define INPUT0_FEATURE_NUM 1000
#define INPUT0_BATCH_NUM 1
#define INPUT0_PAD_BEFORE_SIZE_X 0
#define INPUT0_PAD_BEFORE_SIZE_Y 0
#define INPUT0_PAD_BEFORE_SIZE_Z 0
#define INPUT0_PAD_BEFORE_SIZE_W 0
#define INPUT0_PAD_BEFORE_SIZE_U 0
#define INPUT0_PAD_BEFORE_SIZE_V 0
#define INPUT0_PAD_BEFORE_FEATURE_NUM 0
#define INPUT0_PAD_BEFORE_BATCH_NUM 0
#define INPUT0_PAD_AFTER_SIZE_X 0
#define INPUT0_PAD_AFTER_SIZE_Y 0
#define INPUT0_PAD_AFTER_SIZE_Z 0
#define INPUT0_PAD_AFTER_SIZE_W 0
#define INPUT0_PAD_AFTER_SIZE_U 0
#define INPUT0_PAD_AFTER_SIZE_V 0
#define INPUT0_PAD_AFTER_FEATURE_NUM 0
#define INPUT0_PAD_AFTER_BATCH_NUM 0
#define INPUT0_LENGTH 1000
#define INPUT0_X_PITCH 1
#define INPUT0_Y_PITCH 1
#define INPUT0_Z_PITCH 1
#define INPUT0_W_PITCH 1
#define INPUT0_U_PITCH 1
#define INPUT0_V_PITCH 1
#define INPUT0_FEATURE_PITCH 1
#define INPUT0_BATCH_PITCH 1000
#define INPUT0_GET_INDEX_SAFE(b, f, y, x) ((0 + (f)) % 1000)
#define INPUT0_GET_INDEX(b, f, y, x) (0 + (f))
#define INPUT0_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(INPUT0, b, f, y, x)
#define INPUT0_VIEW_OFFSET 0
#define INPUT0_DIMS 4
#define INPUT0_SIMPLE 1
#define INPUT0_GROUPED 0
#define INPUT0_LAYOUT_BFYX 1
#define INPUT0_TYPE half
#define INPUT0_VAL_MAX HALF_MAX
#define INPUT0_VAL_MIN -INPUT0_VAL_MAX
#define INPUT0_VAL_ONE 1.0h
#define INPUT0_VAL_ZERO 0.0h
#define TO_INPUT0_TYPE(v) convert_half(v)
#define TO_INPUT0_TYPE_SAT(v) convert_half(v)
#define AS_INPUT0_TYPE(v) as_half(v)
#define INPUT0_MAX_FUNC fmax
#define INPUT0_MIN_FUNC fmin
#define INPUT0_ABS_FUNC fabs
#define INPUT0_TYPE_SIZE 2
#define INPUT0_IS_FP 1
#define INPUT0_OFFSET 0
#define INPUT0_SIZES_DATA { 1,1,1000,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(INPUT0_SIZES) = INPUT0_SIZES_DATA;
#define INPUT0_SIZES CONST_ARRAY_REF(INPUT0_SIZES)
#define INPUT0_PITCHES (size_t []){ 1,1,1,1000,1,1,1,1,1, } 
#define INPUT0_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define INPUT0_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OUTPUT_SIZE_X 1
#define OUTPUT_SIZE_Y 1
#define OUTPUT_SIZE_Z 1
#define OUTPUT_SIZE_W 1
#define OUTPUT_SIZE_U 1
#define OUTPUT_SIZE_V 1
#define OUTPUT_FEATURE_NUM 1000
#define OUTPUT_BATCH_NUM 1
#define OUTPUT_PAD_BEFORE_SIZE_X 0
#define OUTPUT_PAD_BEFORE_SIZE_Y 0
#define OUTPUT_PAD_BEFORE_SIZE_Z 0
#define OUTPUT_PAD_BEFORE_SIZE_W 0
#define OUTPUT_PAD_BEFORE_SIZE_U 0
#define OUTPUT_PAD_BEFORE_SIZE_V 0
#define OUTPUT_PAD_BEFORE_FEATURE_NUM 0
#define OUTPUT_PAD_BEFORE_BATCH_NUM 0
#define OUTPUT_PAD_AFTER_SIZE_X 0
#define OUTPUT_PAD_AFTER_SIZE_Y 0
#define OUTPUT_PAD_AFTER_SIZE_Z 0
#define OUTPUT_PAD_AFTER_SIZE_W 0
#define OUTPUT_PAD_AFTER_SIZE_U 0
#define OUTPUT_PAD_AFTER_SIZE_V 0
#define OUTPUT_PAD_AFTER_FEATURE_NUM 0
#define OUTPUT_PAD_AFTER_BATCH_NUM 0
#define OUTPUT_LENGTH 1000
#define OUTPUT_X_PITCH 1
#define OUTPUT_Y_PITCH 1
#define OUTPUT_Z_PITCH 1
#define OUTPUT_W_PITCH 1
#define OUTPUT_U_PITCH 1
#define OUTPUT_V_PITCH 1
#define OUTPUT_FEATURE_PITCH 1
#define OUTPUT_BATCH_PITCH 1000
#define OUTPUT_GET_INDEX_SAFE(b, f, y, x) ((0 + (f)) % 1000)
#define OUTPUT_GET_INDEX(b, f, y, x) (0 + (f))
#define OUTPUT_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(OUTPUT, b, f, y, x)
#define OUTPUT_VIEW_OFFSET 0
#define OUTPUT_DIMS 4
#define OUTPUT_SIMPLE 1
#define OUTPUT_GROUPED 0
#define OUTPUT_LAYOUT_BFYX 1
#define OUTPUT_TYPE float
#define OUTPUT_VAL_MAX FLT_MAX
#define OUTPUT_VAL_MIN -OUTPUT_VAL_MAX
#define OUTPUT_VAL_ONE 1.0f
#define OUTPUT_VAL_ZERO 0.0f
#define TO_OUTPUT_TYPE(v) convert_float(v)
#define TO_OUTPUT_TYPE_SAT(v) convert_float(v)
#define AS_OUTPUT_TYPE(v) as_float(v)
#define OUTPUT_MAX_FUNC fmax
#define OUTPUT_MIN_FUNC fmin
#define OUTPUT_ABS_FUNC fabs
#define OUTPUT_TYPE_SIZE 4
#define OUTPUT_IS_FP 1
#define OUTPUT_OFFSET 0
#define OUTPUT_SIZES_DATA { 1,1,1000,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(OUTPUT_SIZES) = OUTPUT_SIZES_DATA;
#define OUTPUT_SIZES CONST_ARRAY_REF(OUTPUT_SIZES)
#define OUTPUT_PITCHES (size_t []){ 1,1,1,1000,1,1,1,1,1, } 
#define OUTPUT_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OUTPUT_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OPTIONAL_SHAPE_INFO_ARG 
#define OPTIONAL_SHAPE_INFO_TENSOR 
#define MEAN_SUBTRACT_NONE 1
#define CALC_TYPE half
#define CALC_VAL_MAX HALF_MAX
#define CALC_VAL_MIN -CALC_VAL_MAX
#define CALC_VAL_ONE 1.0h
#define CALC_VAL_ZERO 0.0h
#define TO_CALC_TYPE(v) convert_half(v)
#define TO_CALC_TYPE_SAT(v) convert_half(v)
#define AS_CALC_TYPE(v) as_half(v)
#define CALC_MAX_FUNC fmax
#define CALC_MIN_FUNC fmin
#define CALC_ABS_FUNC fabs
#define CALC_TYPE_SIZE 2
#define CALC_IS_FP 1
#define INPUT_REORDER_TYPE half
#define INPUT_REORDER_VAL_MAX HALF_MAX
#define INPUT_REORDER_VAL_MIN -INPUT_REORDER_VAL_MAX
#define INPUT_REORDER_VAL_ONE 1.0h
#define INPUT_REORDER_VAL_ZERO 0.0h
#define TO_INPUT_REORDER_TYPE(v) convert_half(v)
#define TO_INPUT_REORDER_TYPE_SAT(v) convert_half(v)
#define AS_INPUT_REORDER_TYPE(v) as_half(v)
#define INPUT_REORDER_MAX_FUNC fmax
#define INPUT_REORDER_MIN_FUNC fmin
#define INPUT_REORDER_ABS_FUNC fabs
#define INPUT_REORDER_TYPE_SIZE 2
#define INPUT_REORDER_IS_FP 1
#define OUTPUT_REORDER_TYPE float
#define OUTPUT_REORDER_VAL_MAX FLT_MAX
#define OUTPUT_REORDER_VAL_MIN -OUTPUT_REORDER_VAL_MAX
#define OUTPUT_REORDER_VAL_ONE 1.0f
#define OUTPUT_REORDER_VAL_ZERO 0.0f
#define TO_OUTPUT_REORDER_TYPE(v) convert_float(v)
#define TO_OUTPUT_REORDER_TYPE_SAT(v) convert_float(v)
#define AS_OUTPUT_REORDER_TYPE(v) as_float(v)
#define OUTPUT_REORDER_MAX_FUNC fmax
#define OUTPUT_REORDER_MIN_FUNC fmin
#define OUTPUT_REORDER_ABS_FUNC fabs
#define OUTPUT_REORDER_TYPE_SIZE 4
#define OUTPUT_REORDER_IS_FP 1
#define MEAN_OP(val, mean_val) val-mean_val
#define NL_M_TYPED as_float(0x0)/*0.000000e+00*/
#define NL_N_TYPED as_float(0x0)/*0.000000e+00*/
#define ACTIVATION_FUNC_TYPED_TYPE half
#define ACTIVATION_FUNC_TYPED_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_TYPED_VAL_MIN -ACTIVATION_FUNC_TYPED_VAL_MAX
#define ACTIVATION_FUNC_TYPED_VAL_ONE 1.0h
#define ACTIVATION_FUNC_TYPED_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_TYPED_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_TYPED_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_TYPED_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_TYPED_MAX_FUNC fmax
#define ACTIVATION_FUNC_TYPED_MIN_FUNC fmin
#define ACTIVATION_FUNC_TYPED_ABS_FUNC fabs
#define ACTIVATION_FUNC_TYPED_TYPE_SIZE 2
#define ACTIVATION_FUNC_TYPED_IS_FP 1
#define ACTIVATION_PARAMS_TYPED NL_M_TYPED, NL_N_TYPED
#define ACTIVATION_FUNC_TYPED(jit_type, input, m, n) input
#define ACTIVATION_TYPED(jit_type, input, params) ACTIVATION_FUNC_TYPED(jit_type, input, params)
#define SUB_GROUP_SIZE 1
#define GWS_BATCH 2
#define GWS_FEATURE 1
#define GWS_YX 0
#define ELEMENTS_COUNT 1000
#define CHANGE_DATA_TYPE_ONLY 1


inline uint FUNC(get_input_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint v, uint u, uint w, uint z, uint y, uint x) __attribute__((overloadable)) {
#if INPUT0_DIMS < 5
 return INPUT0_GET_INDEX(b, f, y, x);
#elif INPUT0_DIMS == 5
 return INPUT0_GET_INDEX(b, f, z, y, x);
#elif INPUT0_DIMS == 6
 return INPUT0_GET_INDEX(b, f, w, z, y, x);
#elif INPUT0_DIMS == 7
 return INPUT0_GET_INDEX(b, f, u, w, z, y, x);
#elif INPUT0_DIMS == 8
 return INPUT0_GET_INDEX(b, f, v, u, w, z, y, x);
#else
#error [GPU] Unsupported input tensor rank in get_input_index function
#endif
}
inline uint FUNC(get_input_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) __attribute__((overloadable)) {
 return FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, 0, 0, w, z, y, x);
}
inline uint FUNC(get_output_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint v, uint u, uint w, uint z, uint y, uint x) __attribute__((overloadable)) {
#if OUTPUT_DIMS < 5
 return OUTPUT_GET_INDEX(b, f, y, x);
#elif OUTPUT_DIMS == 5
 return OUTPUT_GET_INDEX(b, f, z, y, x);
#elif OUTPUT_DIMS == 6
 return OUTPUT_GET_INDEX(b, f, w, z, y, x);
#elif OUTPUT_DIMS == 7
 return OUTPUT_GET_INDEX(b, f, u, w, z, y, x);
#elif OUTPUT_DIMS == 8
 return OUTPUT_GET_INDEX(b, f, v, u, w, z, y, x);
#else
#error [GPU] Unsupported output tensor rank in get_output_index function
#endif
}
inline uint FUNC(get_output_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) __attribute__((overloadable)) {
 return FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, 0, 0, w, z, y, x);
}
inline uint8 FUNC(reshape_dims)(
 uint b, uint f, uint v, uint u, uint w, uint z, uint y, uint x,
 uint src_size_f, uint src_size_v, uint src_size_u, uint src_size_w, uint src_size_z, uint src_size_y, uint src_size_x,
 uint dst_size_f, uint dst_size_v, uint dst_size_u, uint dst_size_w, uint dst_size_z, uint dst_size_y, uint dst_size_x,
 uint src_dims, uint dst_dims)
{
 if (dst_dims == src_dims) {
 return (uint8)(b, f, v, u, w, z, y, x);
 }
 const uint src_pitch_x = 1;
 const uint src_pitch_y = src_pitch_x * src_size_x;
 const uint src_pitch_z = src_pitch_y * src_size_y;
 const uint src_pitch_w = src_pitch_z * src_size_z;
 const uint src_pitch_u = src_pitch_w * src_size_w;
 const uint src_pitch_v = src_pitch_u * src_size_u;
 const uint src_pitch_f = src_pitch_v * src_size_v;
 const uint src_pitch_b = src_pitch_f * src_size_f;
 uint flat_idx = x * src_pitch_x
 + y * src_pitch_y
 + z * src_pitch_z
 + w * src_pitch_w
 + u * src_pitch_u
 + v * src_pitch_v
 + f * src_pitch_f
 + b * src_pitch_b;
 uint dst_x = flat_idx % dst_size_x;
 flat_idx /= dst_size_x;
 uint dst_y = flat_idx % dst_size_y;
 flat_idx /= dst_size_y;
 uint dst_z = flat_idx % dst_size_z;
 flat_idx /= dst_size_z;
 uint dst_w = flat_idx % dst_size_w;
 flat_idx /= dst_size_w;
 uint dst_u = flat_idx % dst_size_u;
 flat_idx /= dst_size_u;
 uint dst_v = flat_idx % dst_size_v;
 flat_idx /= dst_size_v;
 uint dst_f = flat_idx % dst_size_f;
 flat_idx /= dst_size_f;
 uint dst_b = flat_idx;
 return (uint8)(dst_b, dst_f, dst_v, dst_u, dst_w, dst_z, dst_y, dst_x);
}
#define RESHAPE_DIMS(src_prefix, dst_prefix, b, f, v, u, w, z, y, x) \
 FUNC_CALL(reshape_dims)( \
 b, f, v, u, w, z, y, x, \
 CAT(src_prefix, _FEATURE_NUM), \
 CAT(src_prefix, _SIZE_V), \
 CAT(src_prefix, _SIZE_U), \
 CAT(src_prefix, _SIZE_W), \
 CAT(src_prefix, _SIZE_Z), \
 CAT(src_prefix, _SIZE_Y), \
 CAT(src_prefix, _SIZE_X), \
 CAT(dst_prefix, _FEATURE_NUM), \
 CAT(dst_prefix, _SIZE_V), \
 CAT(dst_prefix, _SIZE_U), \
 CAT(dst_prefix, _SIZE_W), \
 CAT(dst_prefix, _SIZE_Z), \
 CAT(dst_prefix, _SIZE_Y), \
 CAT(dst_prefix, _SIZE_X), \
 CAT(src_prefix, _DIMS), \
 CAT(dst_prefix, _DIMS))
KERNEL (reorder_data_fast_b1)(
 const __global INPUT_REORDER_TYPE* input,
 __global OUTPUT_REORDER_TYPE* output
#ifdef MEAN_SUBTRACT_IN_BUFFER
 , __global MEAN_SUBTRACT_TYPE* mean_subtract
#endif
 )
{
 uint data_idx = get_global_id(0);
 if(data_idx >= ELEMENTS_COUNT)
 return;
#if !CHANGE_DATA_TYPE_ONLY
#if defined OUTPUT_LAYOUT_BFYX
 uint tmp_data_idx = data_idx / OUTPUT_BATCH_NUM;
 const uint b = data_idx - tmp_data_idx * OUTPUT_BATCH_NUM;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_FEATURE_NUM;
 const uint f = data_idx - tmp_data_idx * OUTPUT_FEATURE_NUM;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_X;
 const uint x = data_idx - tmp_data_idx * OUTPUT_SIZE_X;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_Y;
 const uint y = data_idx - tmp_data_idx * OUTPUT_SIZE_Y;
 const uint z = 0;
 const uint w = 0;
#elif defined OUTPUT_LAYOUT_YXFB
 uint tmp_data_idx = data_idx / OUTPUT_SIZE_X;
 const uint x = data_idx - tmp_data_idx * OUTPUT_SIZE_X;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_Y;
 const uint y = data_idx - tmp_data_idx * OUTPUT_SIZE_Y;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_FEATURE_NUM;
 const uint f = data_idx - tmp_data_idx * OUTPUT_FEATURE_NUM;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_BATCH_NUM;
 const uint b = data_idx - tmp_data_idx * OUTPUT_BATCH_NUM;
 const uint z = 0;
 const uint w = 0;
#elif defined OUTPUT_LAYOUT_BFYX_8F
 uint tmp_data_idx = data_idx / OUTPUT_BATCH_NUM;
 const uint b = data_idx - tmp_data_idx * OUTPUT_BATCH_NUM;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_FEATURE_NUM;
 const uint f = data_idx - tmp_data_idx * OUTPUT_FEATURE_NUM;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_X;
 const uint x = data_idx - tmp_data_idx * OUTPUT_SIZE_X;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_Y;
 const uint y = data_idx - tmp_data_idx * OUTPUT_SIZE_Y;
 const uint z = 0;
 const uint w = 0;
#elif defined OUTPUT_LAYOUT_BFYX_16F
 uint tmp_data_idx = data_idx / OUTPUT_BATCH_NUM;
 const uint b = data_idx - tmp_data_idx * OUTPUT_BATCH_NUM;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_FEATURE_NUM;
 const uint f = data_idx - tmp_data_idx * OUTPUT_FEATURE_NUM;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_X;
 const uint x = data_idx - tmp_data_idx * OUTPUT_SIZE_X;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_Y;
 const uint y = data_idx - tmp_data_idx * OUTPUT_SIZE_Y;
 const uint z = 0;
 const uint w = 0;
#elif defined OUTPUT_LAYOUT_BFZYX
 uint tmp_data_idx = data_idx / OUTPUT_BATCH_NUM;
 const uint b = data_idx - tmp_data_idx * OUTPUT_BATCH_NUM;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_FEATURE_NUM;
 const uint f = data_idx - tmp_data_idx * OUTPUT_FEATURE_NUM;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_X;
 const uint x = data_idx - tmp_data_idx * OUTPUT_SIZE_X;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_Y;
 const uint y = data_idx - tmp_data_idx * OUTPUT_SIZE_Y;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_Z;
 const uint z = data_idx - tmp_data_idx * OUTPUT_SIZE_Z;
 const uint w = 0;
#elif defined OUTPUT_LAYOUT_B_FS_ZYX_FSV16
 uint tmp_data_idx = data_idx / OUTPUT_BATCH_NUM;
 const uint b = data_idx - tmp_data_idx * OUTPUT_BATCH_NUM;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_FEATURE_NUM;
 const uint f = data_idx - tmp_data_idx * OUTPUT_FEATURE_NUM;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_X;
 const uint x = data_idx - tmp_data_idx * OUTPUT_SIZE_X;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_Y;
 const uint y = data_idx - tmp_data_idx * OUTPUT_SIZE_Y;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_Z;
 const uint z = data_idx - tmp_data_idx * OUTPUT_SIZE_Z;
 const uint w = 0;
#elif defined OUTPUT_LAYOUT_BFWZYX
 uint tmp_data_idx = data_idx / OUTPUT_BATCH_NUM;
 const uint b = data_idx - tmp_data_idx * OUTPUT_BATCH_NUM;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_FEATURE_NUM;
 const uint f = data_idx - tmp_data_idx * OUTPUT_FEATURE_NUM;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_X;
 const uint x = data_idx - tmp_data_idx * OUTPUT_SIZE_X;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_Y;
 const uint y = data_idx - tmp_data_idx * OUTPUT_SIZE_Y;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_Z;
 const uint z = data_idx - tmp_data_idx * OUTPUT_SIZE_Z;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_W;
 const uint w = data_idx - tmp_data_idx * OUTPUT_SIZE_W;
#else
 uint tmp_data_idx = data_idx / OUTPUT_BATCH_NUM;
 const uint b = data_idx - tmp_data_idx * OUTPUT_BATCH_NUM;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_Y;
 const uint y = data_idx - tmp_data_idx * OUTPUT_SIZE_Y;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_SIZE_X;
 const uint x = data_idx - tmp_data_idx * OUTPUT_SIZE_X;
 data_idx = tmp_data_idx;
 tmp_data_idx = data_idx / OUTPUT_FEATURE_NUM;
 const uint f = data_idx - tmp_data_idx * OUTPUT_FEATURE_NUM;
 const uint z = 0;
 const uint w = 0;
#endif
#endif
#if CHANGE_DATA_TYPE_ONLY
 const uint input_idx = data_idx;
 const uint output_idx = data_idx;
#else
 uint8 ov = RESHAPE_DIMS(OUTPUT, INPUT0, b, f, 0, 0, w, z, y, x);
 const uint input_idx = FUNC_CALL(get_input_index)(ov.s0, ov.s1, ov.s2, ov.s3, ov.s4, ov.s5, ov.s6, ov.s7);
 const uint output_idx = FUNC_CALL(get_output_index)(b, f, 0, 0, w, z, y, x);
#endif
#if defined MEAN_SUBTRACT_INSIDE_PARAMS
 float res = TO_MEAN_TYPE(input[input_idx]);
 res -= VALUE_TO_SUBTRACT[f % VALUE_TO_SUBTRACT_SIZE];
#elif defined MEAN_SUBTRACT_IN_BUFFER
 MEAN_SUBTRACT_TYPE res = TO_MEAN_TYPE(input[input_idx]);
 uint8 msv = RESHAPE_DIMS(INPUT0, MEAN_SUBTRACT, b, f, 0, 0, w, z, y, x);
 res -= mean_subtract[GET_DATA_INDEX_SAFE(MEAN_SUBTRACT, msv.s0, msv.s1, msv.s6, msv.s7)];
#else
 CALC_TYPE res = TO_CALC_TYPE(input[input_idx]);
#endif
 output[output_idx] = ACTIVATION_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE_SAT(res), ACTIVATION_PARAMS_TYPED);
}
#ifdef RESHAPE_DIMS
#undef RESHAPE_DIMS
#endif
#undef KERNEL
#undef KERNEL_ID
#undef FUNC
#undef FUNC_CALL
#undef CONST_ARRAY_DECL
#undef CONST_ARRAY_REF
#ifdef FP64_SUPPORTED
#undef FP64_SUPPORTED
#endif
#ifdef FP16_SUPPORTED
#undef FP16_SUPPORTED
#endif
#ifdef FP16_UNIT_USED
#undef FP16_UNIT_USED
#endif
#ifdef INT8_UNIT_USED
#undef INT8_UNIT_USED
#endif
#ifdef INT32_UNIT_USED
#undef INT32_UNIT_USED
#endif
#ifdef INT64_UNIT_USED
#undef INT64_UNIT_USED
#endif
#ifdef UINT8_UNIT_USED
#undef UINT8_UNIT_USED
#endif
#ifdef UINT32_UNIT_USED
#undef UINT32_UNIT_USED
#endif
#ifdef UNIT_TYPE
#undef UNIT_TYPE
#endif
#ifdef UNIT_VAL_MAX
#undef UNIT_VAL_MAX
#endif
#ifdef UNIT_VAL_MIN
#undef UNIT_VAL_MIN
#endif
#ifdef UNIT_VAL_ONE
#undef UNIT_VAL_ONE
#endif
#ifdef UNIT_VAL_ZERO
#undef UNIT_VAL_ZERO
#endif
#ifdef TO_UNIT_TYPE
#undef TO_UNIT_TYPE
#endif
#ifdef TO_UNIT_TYPE_SAT
#undef TO_UNIT_TYPE_SAT
#endif
#ifdef AS_UNIT_TYPE
#undef AS_UNIT_TYPE
#endif
#ifdef UNIT_MAX_FUNC
#undef UNIT_MAX_FUNC
#endif
#ifdef UNIT_MIN_FUNC
#undef UNIT_MIN_FUNC
#endif
#ifdef UNIT_ABS_FUNC
#undef UNIT_ABS_FUNC
#endif
#ifdef UNIT_TYPE_SIZE
#undef UNIT_TYPE_SIZE
#endif
#ifdef UNIT_IS_FP
#undef UNIT_IS_FP
#endif
#ifdef NL_M
#undef NL_M
#endif
#ifdef NL_N
#undef NL_N
#endif
#ifdef ACTIVATION_FUNC_TYPE
#undef ACTIVATION_FUNC_TYPE
#endif
#ifdef ACTIVATION_FUNC_VAL_MAX
#undef ACTIVATION_FUNC_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_VAL_MIN
#undef ACTIVATION_FUNC_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_VAL_ONE
#undef ACTIVATION_FUNC_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_VAL_ZERO
#undef ACTIVATION_FUNC_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_TYPE
#undef TO_ACTIVATION_FUNC_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_TYPE_SAT
#undef TO_ACTIVATION_FUNC_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_TYPE
#undef AS_ACTIVATION_FUNC_TYPE
#endif
#ifdef ACTIVATION_FUNC_MAX_FUNC
#undef ACTIVATION_FUNC_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_MIN_FUNC
#undef ACTIVATION_FUNC_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_ABS_FUNC
#undef ACTIVATION_FUNC_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_TYPE_SIZE
#undef ACTIVATION_FUNC_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_IS_FP
#undef ACTIVATION_FUNC_IS_FP
#endif
#ifdef ACTIVATION_PARAMS
#undef ACTIVATION_PARAMS
#endif
#ifdef ACTIVATION_FUNC
#undef ACTIVATION_FUNC
#endif
#ifdef ACTIVATION
#undef ACTIVATION
#endif
#ifdef INPUT0_SIZE_X
#undef INPUT0_SIZE_X
#endif
#ifdef INPUT0_SIZE_Y
#undef INPUT0_SIZE_Y
#endif
#ifdef INPUT0_SIZE_Z
#undef INPUT0_SIZE_Z
#endif
#ifdef INPUT0_SIZE_W
#undef INPUT0_SIZE_W
#endif
#ifdef INPUT0_SIZE_U
#undef INPUT0_SIZE_U
#endif
#ifdef INPUT0_SIZE_V
#undef INPUT0_SIZE_V
#endif
#ifdef INPUT0_FEATURE_NUM
#undef INPUT0_FEATURE_NUM
#endif
#ifdef INPUT0_BATCH_NUM
#undef INPUT0_BATCH_NUM
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_X
#undef INPUT0_PAD_BEFORE_SIZE_X
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_Y
#undef INPUT0_PAD_BEFORE_SIZE_Y
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_Z
#undef INPUT0_PAD_BEFORE_SIZE_Z
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_W
#undef INPUT0_PAD_BEFORE_SIZE_W
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_U
#undef INPUT0_PAD_BEFORE_SIZE_U
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_V
#undef INPUT0_PAD_BEFORE_SIZE_V
#endif
#ifdef INPUT0_PAD_BEFORE_FEATURE_NUM
#undef INPUT0_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef INPUT0_PAD_BEFORE_BATCH_NUM
#undef INPUT0_PAD_BEFORE_BATCH_NUM
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_X
#undef INPUT0_PAD_AFTER_SIZE_X
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_Y
#undef INPUT0_PAD_AFTER_SIZE_Y
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_Z
#undef INPUT0_PAD_AFTER_SIZE_Z
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_W
#undef INPUT0_PAD_AFTER_SIZE_W
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_U
#undef INPUT0_PAD_AFTER_SIZE_U
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_V
#undef INPUT0_PAD_AFTER_SIZE_V
#endif
#ifdef INPUT0_PAD_AFTER_FEATURE_NUM
#undef INPUT0_PAD_AFTER_FEATURE_NUM
#endif
#ifdef INPUT0_PAD_AFTER_BATCH_NUM
#undef INPUT0_PAD_AFTER_BATCH_NUM
#endif
#ifdef INPUT0_LENGTH
#undef INPUT0_LENGTH
#endif
#ifdef INPUT0_X_PITCH
#undef INPUT0_X_PITCH
#endif
#ifdef INPUT0_Y_PITCH
#undef INPUT0_Y_PITCH
#endif
#ifdef INPUT0_Z_PITCH
#undef INPUT0_Z_PITCH
#endif
#ifdef INPUT0_W_PITCH
#undef INPUT0_W_PITCH
#endif
#ifdef INPUT0_U_PITCH
#undef INPUT0_U_PITCH
#endif
#ifdef INPUT0_V_PITCH
#undef INPUT0_V_PITCH
#endif
#ifdef INPUT0_FEATURE_PITCH
#undef INPUT0_FEATURE_PITCH
#endif
#ifdef INPUT0_BATCH_PITCH
#undef INPUT0_BATCH_PITCH
#endif
#ifdef INPUT0_GET_INDEX_SAFE
#undef INPUT0_GET_INDEX_SAFE
#endif
#ifdef INPUT0_GET_INDEX
#undef INPUT0_GET_INDEX
#endif
#ifdef INPUT0_GET_INDEX_RAW
#undef INPUT0_GET_INDEX_RAW
#endif
#ifdef INPUT0_VIEW_OFFSET
#undef INPUT0_VIEW_OFFSET
#endif
#ifdef INPUT0_DIMS
#undef INPUT0_DIMS
#endif
#ifdef INPUT0_SIMPLE
#undef INPUT0_SIMPLE
#endif
#ifdef INPUT0_GROUPED
#undef INPUT0_GROUPED
#endif
#ifdef INPUT0_LAYOUT_BFYX
#undef INPUT0_LAYOUT_BFYX
#endif
#ifdef INPUT0_TYPE
#undef INPUT0_TYPE
#endif
#ifdef INPUT0_VAL_MAX
#undef INPUT0_VAL_MAX
#endif
#ifdef INPUT0_VAL_MIN
#undef INPUT0_VAL_MIN
#endif
#ifdef INPUT0_VAL_ONE
#undef INPUT0_VAL_ONE
#endif
#ifdef INPUT0_VAL_ZERO
#undef INPUT0_VAL_ZERO
#endif
#ifdef TO_INPUT0_TYPE
#undef TO_INPUT0_TYPE
#endif
#ifdef TO_INPUT0_TYPE_SAT
#undef TO_INPUT0_TYPE_SAT
#endif
#ifdef AS_INPUT0_TYPE
#undef AS_INPUT0_TYPE
#endif
#ifdef INPUT0_MAX_FUNC
#undef INPUT0_MAX_FUNC
#endif
#ifdef INPUT0_MIN_FUNC
#undef INPUT0_MIN_FUNC
#endif
#ifdef INPUT0_ABS_FUNC
#undef INPUT0_ABS_FUNC
#endif
#ifdef INPUT0_TYPE_SIZE
#undef INPUT0_TYPE_SIZE
#endif
#ifdef INPUT0_IS_FP
#undef INPUT0_IS_FP
#endif
#ifdef INPUT0_OFFSET
#undef INPUT0_OFFSET
#endif
#ifdef INPUT0_SIZES_DATA
#undef INPUT0_SIZES_DATA
#endif
#ifdef INPUT0_PITCHES
#undef INPUT0_PITCHES
#endif
#ifdef INPUT0_PAD_BEFORE
#undef INPUT0_PAD_BEFORE
#endif
#ifdef INPUT0_PAD_AFTER
#undef INPUT0_PAD_AFTER
#endif
#ifdef OUTPUT_SIZE_X
#undef OUTPUT_SIZE_X
#endif
#ifdef OUTPUT_SIZE_Y
#undef OUTPUT_SIZE_Y
#endif
#ifdef OUTPUT_SIZE_Z
#undef OUTPUT_SIZE_Z
#endif
#ifdef OUTPUT_SIZE_W
#undef OUTPUT_SIZE_W
#endif
#ifdef OUTPUT_SIZE_U
#undef OUTPUT_SIZE_U
#endif
#ifdef OUTPUT_SIZE_V
#undef OUTPUT_SIZE_V
#endif
#ifdef OUTPUT_FEATURE_NUM
#undef OUTPUT_FEATURE_NUM
#endif
#ifdef OUTPUT_BATCH_NUM
#undef OUTPUT_BATCH_NUM
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_X
#undef OUTPUT_PAD_BEFORE_SIZE_X
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_Y
#undef OUTPUT_PAD_BEFORE_SIZE_Y
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_Z
#undef OUTPUT_PAD_BEFORE_SIZE_Z
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_W
#undef OUTPUT_PAD_BEFORE_SIZE_W
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_U
#undef OUTPUT_PAD_BEFORE_SIZE_U
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_V
#undef OUTPUT_PAD_BEFORE_SIZE_V
#endif
#ifdef OUTPUT_PAD_BEFORE_FEATURE_NUM
#undef OUTPUT_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef OUTPUT_PAD_BEFORE_BATCH_NUM
#undef OUTPUT_PAD_BEFORE_BATCH_NUM
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_X
#undef OUTPUT_PAD_AFTER_SIZE_X
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_Y
#undef OUTPUT_PAD_AFTER_SIZE_Y
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_Z
#undef OUTPUT_PAD_AFTER_SIZE_Z
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_W
#undef OUTPUT_PAD_AFTER_SIZE_W
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_U
#undef OUTPUT_PAD_AFTER_SIZE_U
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_V
#undef OUTPUT_PAD_AFTER_SIZE_V
#endif
#ifdef OUTPUT_PAD_AFTER_FEATURE_NUM
#undef OUTPUT_PAD_AFTER_FEATURE_NUM
#endif
#ifdef OUTPUT_PAD_AFTER_BATCH_NUM
#undef OUTPUT_PAD_AFTER_BATCH_NUM
#endif
#ifdef OUTPUT_LENGTH
#undef OUTPUT_LENGTH
#endif
#ifdef OUTPUT_X_PITCH
#undef OUTPUT_X_PITCH
#endif
#ifdef OUTPUT_Y_PITCH
#undef OUTPUT_Y_PITCH
#endif
#ifdef OUTPUT_Z_PITCH
#undef OUTPUT_Z_PITCH
#endif
#ifdef OUTPUT_W_PITCH
#undef OUTPUT_W_PITCH
#endif
#ifdef OUTPUT_U_PITCH
#undef OUTPUT_U_PITCH
#endif
#ifdef OUTPUT_V_PITCH
#undef OUTPUT_V_PITCH
#endif
#ifdef OUTPUT_FEATURE_PITCH
#undef OUTPUT_FEATURE_PITCH
#endif
#ifdef OUTPUT_BATCH_PITCH
#undef OUTPUT_BATCH_PITCH
#endif
#ifdef OUTPUT_GET_INDEX_SAFE
#undef OUTPUT_GET_INDEX_SAFE
#endif
#ifdef OUTPUT_GET_INDEX
#undef OUTPUT_GET_INDEX
#endif
#ifdef OUTPUT_GET_INDEX_RAW
#undef OUTPUT_GET_INDEX_RAW
#endif
#ifdef OUTPUT_VIEW_OFFSET
#undef OUTPUT_VIEW_OFFSET
#endif
#ifdef OUTPUT_DIMS
#undef OUTPUT_DIMS
#endif
#ifdef OUTPUT_SIMPLE
#undef OUTPUT_SIMPLE
#endif
#ifdef OUTPUT_GROUPED
#undef OUTPUT_GROUPED
#endif
#ifdef OUTPUT_LAYOUT_BFYX
#undef OUTPUT_LAYOUT_BFYX
#endif
#ifdef OUTPUT_TYPE
#undef OUTPUT_TYPE
#endif
#ifdef OUTPUT_VAL_MAX
#undef OUTPUT_VAL_MAX
#endif
#ifdef OUTPUT_VAL_MIN
#undef OUTPUT_VAL_MIN
#endif
#ifdef OUTPUT_VAL_ONE
#undef OUTPUT_VAL_ONE
#endif
#ifdef OUTPUT_VAL_ZERO
#undef OUTPUT_VAL_ZERO
#endif
#ifdef TO_OUTPUT_TYPE
#undef TO_OUTPUT_TYPE
#endif
#ifdef TO_OUTPUT_TYPE_SAT
#undef TO_OUTPUT_TYPE_SAT
#endif
#ifdef AS_OUTPUT_TYPE
#undef AS_OUTPUT_TYPE
#endif
#ifdef OUTPUT_MAX_FUNC
#undef OUTPUT_MAX_FUNC
#endif
#ifdef OUTPUT_MIN_FUNC
#undef OUTPUT_MIN_FUNC
#endif
#ifdef OUTPUT_ABS_FUNC
#undef OUTPUT_ABS_FUNC
#endif
#ifdef OUTPUT_TYPE_SIZE
#undef OUTPUT_TYPE_SIZE
#endif
#ifdef OUTPUT_IS_FP
#undef OUTPUT_IS_FP
#endif
#ifdef OUTPUT_OFFSET
#undef OUTPUT_OFFSET
#endif
#ifdef OUTPUT_SIZES_DATA
#undef OUTPUT_SIZES_DATA
#endif
#ifdef OUTPUT_PITCHES
#undef OUTPUT_PITCHES
#endif
#ifdef OUTPUT_PAD_BEFORE
#undef OUTPUT_PAD_BEFORE
#endif
#ifdef OUTPUT_PAD_AFTER
#undef OUTPUT_PAD_AFTER
#endif
#ifdef OPTIONAL_SHAPE_INFO_ARG
#undef OPTIONAL_SHAPE_INFO_ARG
#endif
#ifdef OPTIONAL_SHAPE_INFO_TENSOR
#undef OPTIONAL_SHAPE_INFO_TENSOR
#endif
#ifdef MEAN_SUBTRACT_NONE
#undef MEAN_SUBTRACT_NONE
#endif
#ifdef CALC_TYPE
#undef CALC_TYPE
#endif
#ifdef CALC_VAL_MAX
#undef CALC_VAL_MAX
#endif
#ifdef CALC_VAL_MIN
#undef CALC_VAL_MIN
#endif
#ifdef CALC_VAL_ONE
#undef CALC_VAL_ONE
#endif
#ifdef CALC_VAL_ZERO
#undef CALC_VAL_ZERO
#endif
#ifdef TO_CALC_TYPE
#undef TO_CALC_TYPE
#endif
#ifdef TO_CALC_TYPE_SAT
#undef TO_CALC_TYPE_SAT
#endif
#ifdef AS_CALC_TYPE
#undef AS_CALC_TYPE
#endif
#ifdef CALC_MAX_FUNC
#undef CALC_MAX_FUNC
#endif
#ifdef CALC_MIN_FUNC
#undef CALC_MIN_FUNC
#endif
#ifdef CALC_ABS_FUNC
#undef CALC_ABS_FUNC
#endif
#ifdef CALC_TYPE_SIZE
#undef CALC_TYPE_SIZE
#endif
#ifdef CALC_IS_FP
#undef CALC_IS_FP
#endif
#ifdef INPUT_REORDER_TYPE
#undef INPUT_REORDER_TYPE
#endif
#ifdef INPUT_REORDER_VAL_MAX
#undef INPUT_REORDER_VAL_MAX
#endif
#ifdef INPUT_REORDER_VAL_MIN
#undef INPUT_REORDER_VAL_MIN
#endif
#ifdef INPUT_REORDER_VAL_ONE
#undef INPUT_REORDER_VAL_ONE
#endif
#ifdef INPUT_REORDER_VAL_ZERO
#undef INPUT_REORDER_VAL_ZERO
#endif
#ifdef TO_INPUT_REORDER_TYPE
#undef TO_INPUT_REORDER_TYPE
#endif
#ifdef TO_INPUT_REORDER_TYPE_SAT
#undef TO_INPUT_REORDER_TYPE_SAT
#endif
#ifdef AS_INPUT_REORDER_TYPE
#undef AS_INPUT_REORDER_TYPE
#endif
#ifdef INPUT_REORDER_MAX_FUNC
#undef INPUT_REORDER_MAX_FUNC
#endif
#ifdef INPUT_REORDER_MIN_FUNC
#undef INPUT_REORDER_MIN_FUNC
#endif
#ifdef INPUT_REORDER_ABS_FUNC
#undef INPUT_REORDER_ABS_FUNC
#endif
#ifdef INPUT_REORDER_TYPE_SIZE
#undef INPUT_REORDER_TYPE_SIZE
#endif
#ifdef INPUT_REORDER_IS_FP
#undef INPUT_REORDER_IS_FP
#endif
#ifdef OUTPUT_REORDER_TYPE
#undef OUTPUT_REORDER_TYPE
#endif
#ifdef OUTPUT_REORDER_VAL_MAX
#undef OUTPUT_REORDER_VAL_MAX
#endif
#ifdef OUTPUT_REORDER_VAL_MIN
#undef OUTPUT_REORDER_VAL_MIN
#endif
#ifdef OUTPUT_REORDER_VAL_ONE
#undef OUTPUT_REORDER_VAL_ONE
#endif
#ifdef OUTPUT_REORDER_VAL_ZERO
#undef OUTPUT_REORDER_VAL_ZERO
#endif
#ifdef TO_OUTPUT_REORDER_TYPE
#undef TO_OUTPUT_REORDER_TYPE
#endif
#ifdef TO_OUTPUT_REORDER_TYPE_SAT
#undef TO_OUTPUT_REORDER_TYPE_SAT
#endif
#ifdef AS_OUTPUT_REORDER_TYPE
#undef AS_OUTPUT_REORDER_TYPE
#endif
#ifdef OUTPUT_REORDER_MAX_FUNC
#undef OUTPUT_REORDER_MAX_FUNC
#endif
#ifdef OUTPUT_REORDER_MIN_FUNC
#undef OUTPUT_REORDER_MIN_FUNC
#endif
#ifdef OUTPUT_REORDER_ABS_FUNC
#undef OUTPUT_REORDER_ABS_FUNC
#endif
#ifdef OUTPUT_REORDER_TYPE_SIZE
#undef OUTPUT_REORDER_TYPE_SIZE
#endif
#ifdef OUTPUT_REORDER_IS_FP
#undef OUTPUT_REORDER_IS_FP
#endif
#ifdef MEAN_OP
#undef MEAN_OP
#endif
#ifdef NL_M_TYPED
#undef NL_M_TYPED
#endif
#ifdef NL_N_TYPED
#undef NL_N_TYPED
#endif
#ifdef ACTIVATION_FUNC_TYPED_TYPE
#undef ACTIVATION_FUNC_TYPED_TYPE
#endif
#ifdef ACTIVATION_FUNC_TYPED_VAL_MAX
#undef ACTIVATION_FUNC_TYPED_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_TYPED_VAL_MIN
#undef ACTIVATION_FUNC_TYPED_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_TYPED_VAL_ONE
#undef ACTIVATION_FUNC_TYPED_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_TYPED_VAL_ZERO
#undef ACTIVATION_FUNC_TYPED_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_TYPED_TYPE
#undef TO_ACTIVATION_FUNC_TYPED_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_TYPED_TYPE_SAT
#undef TO_ACTIVATION_FUNC_TYPED_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_TYPED_TYPE
#undef AS_ACTIVATION_FUNC_TYPED_TYPE
#endif
#ifdef ACTIVATION_FUNC_TYPED_MAX_FUNC
#undef ACTIVATION_FUNC_TYPED_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_TYPED_MIN_FUNC
#undef ACTIVATION_FUNC_TYPED_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_TYPED_ABS_FUNC
#undef ACTIVATION_FUNC_TYPED_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_TYPED_TYPE_SIZE
#undef ACTIVATION_FUNC_TYPED_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_TYPED_IS_FP
#undef ACTIVATION_FUNC_TYPED_IS_FP
#endif
#ifdef ACTIVATION_PARAMS_TYPED
#undef ACTIVATION_PARAMS_TYPED
#endif
#ifdef ACTIVATION_FUNC_TYPED
#undef ACTIVATION_FUNC_TYPED
#endif
#ifdef ACTIVATION_TYPED
#undef ACTIVATION_TYPED
#endif
#ifdef SUB_GROUP_SIZE
#undef SUB_GROUP_SIZE
#endif
#ifdef GWS_BATCH
#undef GWS_BATCH
#endif
#ifdef GWS_FEATURE
#undef GWS_FEATURE
#endif
#ifdef GWS_YX
#undef GWS_YX
#endif
#ifdef ELEMENTS_COUNT
#undef ELEMENTS_COUNT
#endif
#ifdef CHANGE_DATA_TYPE_ONLY
#undef CHANGE_DATA_TYPE_ONLY
#endif

//====================================================
// Kernel template: convolution_gpu_bfyx_f16_1x1 
// Kernel name: convolution_gpu_bfyx_f16_1x1_17806129452567518898_0_0
#define KERNEL(name) __kernel void convolution_gpu_bfyx_f16_1x1_17806129452567518898_0_0
#define KERNEL_ID convolution_gpu_bfyx_f16_1x1_17806129452567518898_0_0
#define FUNC(name)  _##name##_convolution_gpu_bfyx_f16_1x1_17806129452567518898_0_0
#define FUNC_CALL(name)  _##name##_convolution_gpu_bfyx_f16_1x1_17806129452567518898_0_0
#define CONST_ARRAY_DECL(name) __constant size_t  _##name##_convolution_gpu_bfyx_f16_1x1_17806129452567518898_0_0 []
#define CONST_ARRAY_REF(name)  _##name##_convolution_gpu_bfyx_f16_1x1_17806129452567518898_0_0
#define FP64_SUPPORTED 0
#define FP16_SUPPORTED 1
#define FP16_UNIT_USED 1
#define INT8_UNIT_USED 0
#define INT32_UNIT_USED 0
#define INT64_UNIT_USED 0
#define UINT8_UNIT_USED 0
#define UINT32_UNIT_USED 0
#define UNIT_TYPE half
#define UNIT_VAL_MAX HALF_MAX
#define UNIT_VAL_MIN -UNIT_VAL_MAX
#define UNIT_VAL_ONE 1.0h
#define UNIT_VAL_ZERO 0.0h
#define TO_UNIT_TYPE(v) convert_half(v)
#define TO_UNIT_TYPE_SAT(v) convert_half(v)
#define AS_UNIT_TYPE(v) as_half(v)
#define UNIT_MAX_FUNC fmax
#define UNIT_MIN_FUNC fmin
#define UNIT_ABS_FUNC fabs
#define UNIT_TYPE_SIZE 2
#define UNIT_IS_FP 1
#define NL_M as_float(0x0)/*0.000000e+00*/
#define NL_N as_float(0x0)/*0.000000e+00*/
#define ACTIVATION_FUNC_TYPE half
#define ACTIVATION_FUNC_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_VAL_MIN -ACTIVATION_FUNC_VAL_MAX
#define ACTIVATION_FUNC_VAL_ONE 1.0h
#define ACTIVATION_FUNC_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_MAX_FUNC fmax
#define ACTIVATION_FUNC_MIN_FUNC fmin
#define ACTIVATION_FUNC_ABS_FUNC fabs
#define ACTIVATION_FUNC_TYPE_SIZE 2
#define ACTIVATION_FUNC_IS_FP 1
#define ACTIVATION_PARAMS NL_M, NL_N
#define ACTIVATION_FUNC(input, m, n) input
#define ACTIVATION(input, params) ACTIVATION_FUNC(input, params)
#define INPUT0_SIZE_X 56
#define INPUT0_SIZE_Y 56
#define INPUT0_SIZE_Z 1
#define INPUT0_SIZE_W 1
#define INPUT0_SIZE_U 1
#define INPUT0_SIZE_V 1
#define INPUT0_FEATURE_NUM 256
#define INPUT0_BATCH_NUM 1
#define INPUT0_PAD_BEFORE_SIZE_X 0
#define INPUT0_PAD_BEFORE_SIZE_Y 0
#define INPUT0_PAD_BEFORE_SIZE_Z 0
#define INPUT0_PAD_BEFORE_SIZE_W 0
#define INPUT0_PAD_BEFORE_SIZE_U 0
#define INPUT0_PAD_BEFORE_SIZE_V 0
#define INPUT0_PAD_BEFORE_FEATURE_NUM 0
#define INPUT0_PAD_BEFORE_BATCH_NUM 0
#define INPUT0_PAD_AFTER_SIZE_X 0
#define INPUT0_PAD_AFTER_SIZE_Y 0
#define INPUT0_PAD_AFTER_SIZE_Z 0
#define INPUT0_PAD_AFTER_SIZE_W 0
#define INPUT0_PAD_AFTER_SIZE_U 0
#define INPUT0_PAD_AFTER_SIZE_V 0
#define INPUT0_PAD_AFTER_FEATURE_NUM 0
#define INPUT0_PAD_AFTER_BATCH_NUM 0
#define INPUT0_LENGTH 802816
#define INPUT0_X_PITCH 1
#define INPUT0_Y_PITCH 56
#define INPUT0_Z_PITCH 1
#define INPUT0_W_PITCH 1
#define INPUT0_U_PITCH 1
#define INPUT0_V_PITCH 1
#define INPUT0_FEATURE_PITCH 3136
#define INPUT0_BATCH_PITCH 802816
#define INPUT0_GET_INDEX_SAFE(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX_SAFE(INPUT0, b, f, y, x)
#define INPUT0_GET_INDEX(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(INPUT0, b, f, y, x)
#define INPUT0_GET_INDEX_RAW(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(INPUT0, b, f, y, x)
#define INPUT0_VIEW_OFFSET 0
#define INPUT0_DIMS 4
#define INPUT0_SIMPLE 0
#define INPUT0_GROUPED 0
#define INPUT0_LAYOUT_B_FS_YX_FSV16 1
#define INPUT0_TYPE half
#define INPUT0_VAL_MAX HALF_MAX
#define INPUT0_VAL_MIN -INPUT0_VAL_MAX
#define INPUT0_VAL_ONE 1.0h
#define INPUT0_VAL_ZERO 0.0h
#define TO_INPUT0_TYPE(v) convert_half(v)
#define TO_INPUT0_TYPE_SAT(v) convert_half(v)
#define AS_INPUT0_TYPE(v) as_half(v)
#define INPUT0_MAX_FUNC fmax
#define INPUT0_MIN_FUNC fmin
#define INPUT0_ABS_FUNC fabs
#define INPUT0_TYPE_SIZE 2
#define INPUT0_IS_FP 1
#define INPUT0_OFFSET 0
#define INPUT0_SIZES_DATA { 56,56,256,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(INPUT0_SIZES) = INPUT0_SIZES_DATA;
#define INPUT0_SIZES CONST_ARRAY_REF(INPUT0_SIZES)
#define INPUT0_PITCHES (size_t []){ 1,56,3136,802816,1,1,1,1,1, } 
#define INPUT0_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define INPUT0_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OUTPUT_SIZE_X 56
#define OUTPUT_SIZE_Y 56
#define OUTPUT_SIZE_Z 1
#define OUTPUT_SIZE_W 1
#define OUTPUT_SIZE_U 1
#define OUTPUT_SIZE_V 1
#define OUTPUT_FEATURE_NUM 128
#define OUTPUT_BATCH_NUM 1
#define OUTPUT_PAD_BEFORE_SIZE_X 1
#define OUTPUT_PAD_BEFORE_SIZE_Y 1
#define OUTPUT_PAD_BEFORE_SIZE_Z 0
#define OUTPUT_PAD_BEFORE_SIZE_W 0
#define OUTPUT_PAD_BEFORE_SIZE_U 0
#define OUTPUT_PAD_BEFORE_SIZE_V 0
#define OUTPUT_PAD_BEFORE_FEATURE_NUM 0
#define OUTPUT_PAD_BEFORE_BATCH_NUM 0
#define OUTPUT_PAD_AFTER_SIZE_X 1
#define OUTPUT_PAD_AFTER_SIZE_Y 1
#define OUTPUT_PAD_AFTER_SIZE_Z 0
#define OUTPUT_PAD_AFTER_SIZE_W 0
#define OUTPUT_PAD_AFTER_SIZE_U 0
#define OUTPUT_PAD_AFTER_SIZE_V 0
#define OUTPUT_PAD_AFTER_FEATURE_NUM 0
#define OUTPUT_PAD_AFTER_BATCH_NUM 0
#define OUTPUT_LENGTH 401408
#define OUTPUT_X_PITCH 1
#define OUTPUT_Y_PITCH 58
#define OUTPUT_Z_PITCH 1
#define OUTPUT_W_PITCH 1
#define OUTPUT_U_PITCH 1
#define OUTPUT_V_PITCH 1
#define OUTPUT_FEATURE_PITCH 3364
#define OUTPUT_BATCH_PITCH 430592
#define OUTPUT_GET_INDEX_SAFE(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX_SAFE(OUTPUT, b, f, y, x)
#define OUTPUT_GET_INDEX(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(OUTPUT, b, f, y, x)
#define OUTPUT_GET_INDEX_RAW(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(OUTPUT, b, f, y, x)
#define OUTPUT_VIEW_OFFSET 0
#define OUTPUT_DIMS 4
#define OUTPUT_SIMPLE 0
#define OUTPUT_GROUPED 0
#define OUTPUT_LAYOUT_B_FS_YX_FSV16 1
#define OUTPUT_TYPE half
#define OUTPUT_VAL_MAX HALF_MAX
#define OUTPUT_VAL_MIN -OUTPUT_VAL_MAX
#define OUTPUT_VAL_ONE 1.0h
#define OUTPUT_VAL_ZERO 0.0h
#define TO_OUTPUT_TYPE(v) convert_half(v)
#define TO_OUTPUT_TYPE_SAT(v) convert_half(v)
#define AS_OUTPUT_TYPE(v) as_half(v)
#define OUTPUT_MAX_FUNC fmax
#define OUTPUT_MIN_FUNC fmin
#define OUTPUT_ABS_FUNC fabs
#define OUTPUT_TYPE_SIZE 2
#define OUTPUT_IS_FP 1
#define OUTPUT_OFFSET 59
#define OUTPUT_SIZES_DATA { 56,56,128,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(OUTPUT_SIZES) = OUTPUT_SIZES_DATA;
#define OUTPUT_SIZES CONST_ARRAY_REF(OUTPUT_SIZES)
#define OUTPUT_PITCHES (size_t []){ 1,58,3364,430592,1,1,1,1,1, } 
#define OUTPUT_PAD_BEFORE (size_t []){ 1,1,0,0,0,0,0,0,0, } 
#define OUTPUT_PAD_AFTER (size_t []){ 1,1,0,0,0,0,0,0,0, } 
#define OPTIONAL_SHAPE_INFO_ARG 
#define OPTIONAL_SHAPE_INFO_TENSOR 
#define FILTER_SIZE_X 1
#define FILTER_SIZE_Y 1
#define FILTER_SIZE_Z 1
#define FILTER_IFM_NUM 256
#define FILTER_OFM_NUM 128
#define FILTER_GROUPS_NUM 1
#define FILTER_X_PITCH 1
#define FILTER_Y_PITCH 1
#define FILTER_Z_PITCH 1
#define FILTER_IFM_PITCH 1
#define FILTER_OFM_PITCH 256
#define FILTER_GROUPS_PITCH 1
#define FILTER_LENGTH 32768
#define FILTER_VIEW_OFFSET 0
#define FILTER_DIMS 4
#define FILTER_SIMPLE 0
#define FILTER_GROUPED 0
#define FILTER_LAYOUT_OS_IS_YX_ISV16_OSV16 1
#define FILTER_TYPE half
#define FILTER_VAL_MAX HALF_MAX
#define FILTER_VAL_MIN -FILTER_VAL_MAX
#define FILTER_VAL_ONE 1.0h
#define FILTER_VAL_ZERO 0.0h
#define TO_FILTER_TYPE(v) convert_half(v)
#define TO_FILTER_TYPE_SAT(v) convert_half(v)
#define AS_FILTER_TYPE(v) as_half(v)
#define FILTER_MAX_FUNC fmax
#define FILTER_MIN_FUNC fmin
#define FILTER_ABS_FUNC fabs
#define FILTER_TYPE_SIZE 2
#define FILTER_IS_FP 1
#define FILTER_OFFSET 0
#define FILTER_SIZES_DATA { 1,1,256,128,1,1,1,1,1, } 
CONST_ARRAY_DECL(FILTER_SIZES) = FILTER_SIZES_DATA;
#define FILTER_SIZES CONST_ARRAY_REF(FILTER_SIZES)
#define FILTER_PITCHES (size_t []){ 1,1,1,256,1,1,1,1,1, } 
#define FILTER_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FILTER_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define BIAS_TERM 1
#define BIAS_SIZE_X 1
#define BIAS_SIZE_Y 1
#define BIAS_SIZE_Z 1
#define BIAS_SIZE_W 1
#define BIAS_SIZE_U 1
#define BIAS_SIZE_V 1
#define BIAS_FEATURE_NUM 128
#define BIAS_BATCH_NUM 1
#define BIAS_PAD_BEFORE_SIZE_X 0
#define BIAS_PAD_BEFORE_SIZE_Y 0
#define BIAS_PAD_BEFORE_SIZE_Z 0
#define BIAS_PAD_BEFORE_SIZE_W 0
#define BIAS_PAD_BEFORE_SIZE_U 0
#define BIAS_PAD_BEFORE_SIZE_V 0
#define BIAS_PAD_BEFORE_FEATURE_NUM 0
#define BIAS_PAD_BEFORE_BATCH_NUM 0
#define BIAS_PAD_AFTER_SIZE_X 0
#define BIAS_PAD_AFTER_SIZE_Y 0
#define BIAS_PAD_AFTER_SIZE_Z 0
#define BIAS_PAD_AFTER_SIZE_W 0
#define BIAS_PAD_AFTER_SIZE_U 0
#define BIAS_PAD_AFTER_SIZE_V 0
#define BIAS_PAD_AFTER_FEATURE_NUM 0
#define BIAS_PAD_AFTER_BATCH_NUM 0
#define BIAS_LENGTH 128
#define BIAS_X_PITCH 1
#define BIAS_Y_PITCH 1
#define BIAS_Z_PITCH 1
#define BIAS_W_PITCH 1
#define BIAS_U_PITCH 1
#define BIAS_V_PITCH 1
#define BIAS_FEATURE_PITCH 1
#define BIAS_BATCH_PITCH 128
#define BIAS_GET_INDEX_SAFE(b, f, y, x) ((0 + (f)) % 128)
#define BIAS_GET_INDEX(b, f, y, x) (0 + (f))
#define BIAS_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(BIAS, b, f, y, x)
#define BIAS_VIEW_OFFSET 0
#define BIAS_DIMS 2
#define BIAS_SIMPLE 1
#define BIAS_GROUPED 0
#define BIAS_LAYOUT_BF 1
#define BIAS_TYPE half
#define BIAS_VAL_MAX HALF_MAX
#define BIAS_VAL_MIN -BIAS_VAL_MAX
#define BIAS_VAL_ONE 1.0h
#define BIAS_VAL_ZERO 0.0h
#define TO_BIAS_TYPE(v) convert_half(v)
#define TO_BIAS_TYPE_SAT(v) convert_half(v)
#define AS_BIAS_TYPE(v) as_half(v)
#define BIAS_MAX_FUNC fmax
#define BIAS_MIN_FUNC fmin
#define BIAS_ABS_FUNC fabs
#define BIAS_TYPE_SIZE 2
#define BIAS_IS_FP 1
#define BIAS_OFFSET 0
#define BIAS_SIZES_DATA { 128,1,1,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(BIAS_SIZES) = BIAS_SIZES_DATA;
#define BIAS_SIZES CONST_ARRAY_REF(BIAS_SIZES)
#define BIAS_PITCHES (size_t []){ 1,128,1,1,1,1,1,1,1, } 
#define BIAS_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define BIAS_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define BIAS_PER_OUTPUT 0
#define BIAS_PER_OFM 1
#define STRIDE_SIZE_X 1
#define STRIDE_SIZE_Y 1
#define STRIDE_SIZE_Z 1
#define PADDING_SIZE_X 0
#define PADDING_SIZE_Y 0
#define PADDING_SIZE_Z 0
#define DILATION_SIZE_X 1
#define DILATION_SIZE_Y 1
#define DILATION_SIZE_Z 1
#define FILTER_ARRAY_NUM 1
#define INPUT0_OFFSET_WITH_PADDING 0
#define GROUPED 0
#define SKIP_BATCH 1
#define FUSED_OP0_LOAD_VEC 
#define ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE half
#define ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_MIN -ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_MAX
#define ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_ONE 1.0h
#define ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_FUSED_OP0_VEC_MAX_FUNC fmax
#define ACTIVATION_FUNC_FUSED_OP0_VEC_MIN_FUNC fmin
#define ACTIVATION_FUNC_FUSED_OP0_VEC_ABS_FUNC fabs
#define ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE_SIZE 2
#define ACTIVATION_FUNC_FUSED_OP0_VEC_IS_FP 1
#define ACTIVATION_PARAMS_FUSED_OP0_VEC NL_M_FUSED_OP0_VEC, NL_N_FUSED_OP0_VEC
#define ACTIVATION_FUNC_FUSED_OP0_VEC(input, m, n) (ACTIVATION_FUNC_FUSED_OP0_VEC_MAX_FUNC(ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_ZERO, input))
#define ACTIVATION_FUSED_OP0_VEC(input, params) ACTIVATION_FUNC_FUSED_OP0_VEC(input, params)
#define FUSED_OP0_ACTION_VEC \
	half8 dst_out_0 = convert_half8(dst);\
	dst_out_0 = ACTIVATION_FUNC_FUSED_OP0_VEC(dst_out_0, (half8)(as_float(0x0)/*0.000000e+00*/),(half8)(as_float(0x0)/*0.000000e+00*/));
#define FUSED_OPS_VEC \
	FUSED_OP0_LOAD_VEC\
	FUSED_OP0_ACTION_VEC
#define FUSED_OPS_PRELOAD_VEC \
	FUSED_OP0_LOAD_VEC
#define FUSED_OPS_CALC_VEC \
	FUSED_OP0_ACTION_VEC
#define FUSED_OPS_RESULT_VEC dst_out_0
#define FUSED_OPS_CAN_USE_PRELOAD_VEC 1
#define FUSED_OP0_LOAD_SCALAR 
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE half
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_MIN -ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_MAX
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_ONE 1.0h
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_MAX_FUNC fmax
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_MIN_FUNC fmin
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_ABS_FUNC fabs
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE_SIZE 2
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_IS_FP 1
#define ACTIVATION_PARAMS_FUSED_OP0_SCALAR NL_M_FUSED_OP0_SCALAR, NL_N_FUSED_OP0_SCALAR
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR(input, m, n) (ACTIVATION_FUNC_FUSED_OP0_SCALAR_MAX_FUNC(ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_ZERO, input))
#define ACTIVATION_FUSED_OP0_SCALAR(input, params) ACTIVATION_FUNC_FUSED_OP0_SCALAR(input, params)
#define FUSED_OP0_ACTION_SCALAR \
	half dst_i__out_0 = convert_half(dst[i]);\
	dst_i__out_0 = ACTIVATION_FUNC_FUSED_OP0_SCALAR(dst_i__out_0, (half)(as_float(0x0)/*0.000000e+00*/),(half)(as_float(0x0)/*0.000000e+00*/));
#define FUSED_OPS_SCALAR \
	FUSED_OP0_LOAD_SCALAR\
	FUSED_OP0_ACTION_SCALAR
#define FUSED_OPS_PRELOAD_SCALAR \
	FUSED_OP0_LOAD_SCALAR
#define FUSED_OPS_CALC_SCALAR \
	FUSED_OP0_ACTION_SCALAR
#define FUSED_OPS_RESULT_SCALAR dst_i__out_0
#define FUSED_OPS_CAN_USE_PRELOAD_SCALAR 1
#define FUSED_OP0_LOAD_SCALAR_B1 
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_TYPE half
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_VAL_MIN -ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_VAL_MAX
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_VAL_ONE 1.0h
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_MAX_FUNC fmax
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_MIN_FUNC fmin
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_ABS_FUNC fabs
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_TYPE_SIZE 2
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_IS_FP 1
#define ACTIVATION_PARAMS_FUSED_OP0_SCALAR_B1 NL_M_FUSED_OP0_SCALAR_B1, NL_N_FUSED_OP0_SCALAR_B1
#define ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1(input, m, n) (ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_MAX_FUNC(ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_VAL_ZERO, input))
#define ACTIVATION_FUSED_OP0_SCALAR_B1(input, params) ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1(input, params)
#define FUSED_OP0_ACTION_SCALAR_B1 \
	half dst_out_0 = convert_half(dst);\
	dst_out_0 = ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1(dst_out_0, (half)(as_float(0x0)/*0.000000e+00*/),(half)(as_float(0x0)/*0.000000e+00*/));
#define FUSED_OPS_SCALAR_B1 \
	FUSED_OP0_LOAD_SCALAR_B1\
	FUSED_OP0_ACTION_SCALAR_B1
#define FUSED_OPS_PRELOAD_SCALAR_B1 \
	FUSED_OP0_LOAD_SCALAR_B1
#define FUSED_OPS_CALC_SCALAR_B1 \
	FUSED_OP0_ACTION_SCALAR_B1
#define FUSED_OPS_RESULT_SCALAR_B1 dst_out_0
#define FUSED_OPS_CAN_USE_PRELOAD_SCALAR_B1 1
#define FUSED_OP_0_OUTPUT_SIZE_X 56
#define FUSED_OP_0_OUTPUT_SIZE_Y 56
#define FUSED_OP_0_OUTPUT_SIZE_Z 1
#define FUSED_OP_0_OUTPUT_SIZE_W 1
#define FUSED_OP_0_OUTPUT_SIZE_U 1
#define FUSED_OP_0_OUTPUT_SIZE_V 1
#define FUSED_OP_0_OUTPUT_FEATURE_NUM 128
#define FUSED_OP_0_OUTPUT_BATCH_NUM 1
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_X 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Y 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Z 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_W 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_U 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_V 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_FEATURE_NUM 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_BATCH_NUM 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_X 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Y 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Z 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_W 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_U 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_V 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_FEATURE_NUM 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_BATCH_NUM 0
#define FUSED_OP_0_OUTPUT_LENGTH 401408
#define FUSED_OP_0_OUTPUT_X_PITCH 1
#define FUSED_OP_0_OUTPUT_Y_PITCH 56
#define FUSED_OP_0_OUTPUT_Z_PITCH 1
#define FUSED_OP_0_OUTPUT_W_PITCH 1
#define FUSED_OP_0_OUTPUT_U_PITCH 1
#define FUSED_OP_0_OUTPUT_V_PITCH 1
#define FUSED_OP_0_OUTPUT_FEATURE_PITCH 3136
#define FUSED_OP_0_OUTPUT_BATCH_PITCH 401408
#define FUSED_OP_0_OUTPUT_GET_INDEX_SAFE(b, f, y, x) GET_DATA_INDEX_SAFE(FUSED_OP_0_OUTPUT, b, f, y, x)
#define FUSED_OP_0_OUTPUT_GET_INDEX(b, f, y, x) GET_DATA_INDEX(FUSED_OP_0_OUTPUT, b, f, y, x)
#define FUSED_OP_0_OUTPUT_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(FUSED_OP_0_OUTPUT, b, f, y, x)
#define FUSED_OP_0_OUTPUT_VIEW_OFFSET 0
#define FUSED_OP_0_OUTPUT_DIMS 4
#define FUSED_OP_0_OUTPUT_SIMPLE 1
#define FUSED_OP_0_OUTPUT_GROUPED 0
#define FUSED_OP_0_OUTPUT_LAYOUT_BFYX 1
#define FUSED_OP_0_OUTPUT_TYPE half
#define FUSED_OP_0_OUTPUT_VAL_MAX HALF_MAX
#define FUSED_OP_0_OUTPUT_VAL_MIN -FUSED_OP_0_OUTPUT_VAL_MAX
#define FUSED_OP_0_OUTPUT_VAL_ONE 1.0h
#define FUSED_OP_0_OUTPUT_VAL_ZERO 0.0h
#define TO_FUSED_OP_0_OUTPUT_TYPE(v) convert_half(v)
#define TO_FUSED_OP_0_OUTPUT_TYPE_SAT(v) convert_half(v)
#define AS_FUSED_OP_0_OUTPUT_TYPE(v) as_half(v)
#define FUSED_OP_0_OUTPUT_MAX_FUNC fmax
#define FUSED_OP_0_OUTPUT_MIN_FUNC fmin
#define FUSED_OP_0_OUTPUT_ABS_FUNC fabs
#define FUSED_OP_0_OUTPUT_TYPE_SIZE 2
#define FUSED_OP_0_OUTPUT_IS_FP 1
#define FUSED_OP_0_OUTPUT_OFFSET 0
#define FUSED_OP_0_OUTPUT_SIZES_DATA { 56,56,128,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(FUSED_OP_0_OUTPUT_SIZES) = FUSED_OP_0_OUTPUT_SIZES_DATA;
#define FUSED_OP_0_OUTPUT_SIZES CONST_ARRAY_REF(FUSED_OP_0_OUTPUT_SIZES)
#define FUSED_OP_0_OUTPUT_PITCHES (size_t []){ 1,56,3136,401408,1,1,1,1,1, } 
#define FUSED_OP_0_OUTPUT_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP_0_OUTPUT_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP0_DECLS 
#define FUSED_OP0_ARGS 
#define FUSED_OPS_DECLS 
#define FUSED_OPS_ARGS 
#define HAS_FUSED_OPS 1
#define HAS_FUSED_OPS_DECLS 0
#define X_BLOCK_SIZE 8
#define SLM_DIV_FACTOR 1
#define SUB_GROUP_SIZE 16
#define WORK_GROUP_SIZE 16
#define PADDED_INPUT 0
#define PADDED_OUTPUT 1
#define NON_UNIT_FUSED_OP_SPATIAL 0
#define IC_BLOCKS 16
#define INPUT_LEFTOVERS 0


#ifndef UNIT_TYPE
#error Unit type not defined.
#endif
#if UNIT_TYPE_SIZE == 2
#define UNIT_BLOCK_RW_TYPE ushort
#define UNIT_BLOCK_READ_FUNC _sub_group_block_read_us
#define UNIT_BLOCK_WRITE_FUNC _sub_group_block_write_us
#elif UNIT_TYPE_SIZE == 4
#define UNIT_BLOCK_RW_TYPE uint
#define UNIT_BLOCK_READ_FUNC _sub_group_block_read
#define UNIT_BLOCK_WRITE_FUNC _sub_group_block_write
#else
#error Unsupported unit type for block read/write.
#endif
#define UNIT_TYPE2 MAKE_VECTOR_TYPE(UNIT_TYPE, 2)
#define UNIT_TYPE4 MAKE_VECTOR_TYPE(UNIT_TYPE, 4)
#define UNIT_TYPE8 MAKE_VECTOR_TYPE(UNIT_TYPE, 8)
#define UNIT_BLOCK_RW_TYPE2 MAKE_VECTOR_TYPE(UNIT_BLOCK_RW_TYPE, 2)
#define UNIT_BLOCK_RW_TYPE4 MAKE_VECTOR_TYPE(UNIT_BLOCK_RW_TYPE, 4)
#define UNIT_BLOCK_RW_TYPE8 MAKE_VECTOR_TYPE(UNIT_BLOCK_RW_TYPE, 8)
#define UNIT_BLOCK_READ_FUNC2 CAT(UNIT_BLOCK_READ_FUNC, 2)
#define UNIT_BLOCK_READ_FUNC4 CAT(UNIT_BLOCK_READ_FUNC, 4)
#define UNIT_BLOCK_READ_FUNC8 CAT(UNIT_BLOCK_READ_FUNC, 8)
#define UNIT_BLOCK_WRITE_FUNC2 CAT(UNIT_BLOCK_WRITE_FUNC, 2)
#define UNIT_BLOCK_WRITE_FUNC4 CAT(UNIT_BLOCK_WRITE_FUNC, 4)
#define UNIT_BLOCK_WRITE_FUNC8 CAT(UNIT_BLOCK_WRITE_FUNC, 8)
#define UNIT_BLOCK_READ(ptr, offset) AS_TYPE(UNIT_TYPE, UNIT_BLOCK_READ_FUNC( (const __global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset)))
#define UNIT_BLOCK_READ2(ptr, offset) AS_TYPE(UNIT_TYPE2, UNIT_BLOCK_READ_FUNC2((const __global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset)))
#define UNIT_BLOCK_READ4(ptr, offset) AS_TYPE(UNIT_TYPE4, UNIT_BLOCK_READ_FUNC4((const __global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset)))
#define UNIT_BLOCK_READ8(ptr, offset) AS_TYPE(UNIT_TYPE8, UNIT_BLOCK_READ_FUNC8((const __global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset)))
#define UNIT_BLOCK_WRITE(ptr, offset, val) UNIT_BLOCK_WRITE_FUNC( (__global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset), AS_TYPE(UNIT_BLOCK_RW_TYPE, val))
#define UNIT_BLOCK_WRITE2(ptr, offset, val) UNIT_BLOCK_WRITE_FUNC2((__global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset), AS_TYPE(UNIT_BLOCK_RW_TYPE2, val))
#define UNIT_BLOCK_WRITE4(ptr, offset, val) UNIT_BLOCK_WRITE_FUNC4((__global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset), AS_TYPE(UNIT_BLOCK_RW_TYPE4, val))
#define UNIT_BLOCK_WRITE8(ptr, offset, val) UNIT_BLOCK_WRITE_FUNC8((__global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset), AS_TYPE(UNIT_BLOCK_RW_TYPE8, val))
#if X_BLOCK_SIZE > 1
# define GET_SRC(data, id) AS_TYPE(MAKE_VECTOR_TYPE(UNIT_TYPE, X_BLOCK_SIZE), \
 _sub_group_shuffle( \
 AS_TYPE(MAKE_VECTOR_TYPE(UNIT_BLOCK_RW_TYPE, X_BLOCK_SIZE), data), \
 id))
#else
# define GET_SRC(data, id) AS_TYPE(UNIT_TYPE, _sub_group_shuffle(AS_TYPE(UNIT_BLOCK_RW_TYPE, data), id))
#endif
#define FEATURE_SLICE_SIZE 16
#if X_BLOCK_SIZE > 1
# define UNIT_BLOCK_READ_VEC(ptr, offset) CAT(UNIT_BLOCK_READ, X_BLOCK_SIZE)(ptr, offset)
# define UNIT_BLOCK_WRITE_VEC(ptr, offset, val) CAT(UNIT_BLOCK_WRITE, X_BLOCK_SIZE)(ptr, offset, val)
#endif
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, SUB_GROUP_SIZE * SLM_DIV_FACTOR, 1)))
KERNEL(convolution_b_fs_yx_fsv16_1x1)(
 OPTIONAL_SHAPE_INFO_ARG
 __global INPUT0_TYPE* input,
 __global OUTPUT_TYPE* output,
 __global FILTER_TYPE* weights
#if BIAS_TERM
 , __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
 , FUSED_OPS_DECLS
#endif
) {
#if X_BLOCK_SIZE > 1
 const uint xy = (int)get_global_id(0);
 const uint x = (xy * X_BLOCK_SIZE) % OUTPUT_SIZE_X;
 const uint y = (xy * X_BLOCK_SIZE) / OUTPUT_SIZE_X;
 const uint input_x = x;
 const uint input_y = y;
#endif
 const uint b = (int)get_global_id(2);
 const uint sglid = (int)get_sub_group_local_id();
 const uint lid1 = (int)get_local_id(1);
 const uint feature_per_wg = (int)get_local_size(1) / SLM_DIV_FACTOR;
 const uint feature_sub_block = lid1 / feature_per_wg;
 const uint feature_block = (int)get_group_id(1);
 const uint input_x_pitch = FEATURE_SLICE_SIZE;
 const uint input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
 const uint input_fs_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
 const uint input_total_f_size = INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
 const uint input_b_pitch = input_fs_pitch * ((input_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
 const uint input_fs_pad_before = INPUT0_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;
 const uint input_offset = b * input_b_pitch +
 input_fs_pad_before * input_fs_pitch +
 INPUT0_PAD_BEFORE_SIZE_Y * input_y_pitch +
 INPUT0_PAD_BEFORE_SIZE_X * input_x_pitch;
 const uint output_x_pitch = FEATURE_SLICE_SIZE;
 const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
 const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
 const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
 const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
 const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;
 const uint output_offset = b * output_b_pitch +
 (feature_block + output_fs_pad_before) * output_fs_pitch +
 (OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
 (OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;
 const uint filter_isv_pitch = FEATURE_SLICE_SIZE;
 const uint filter_x_pitch = FEATURE_SLICE_SIZE * FEATURE_SLICE_SIZE;
 const uint filter_y_pitch = filter_x_pitch * FILTER_SIZE_X;
 const uint filter_is_pitch = filter_y_pitch * FILTER_SIZE_Y;
 const uint filter_os_pitch = filter_is_pitch * ((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
 const uint filter_offset = feature_block * filter_os_pitch;
#if X_BLOCK_SIZE > 1
 typedef MAKE_VECTOR_TYPE(UNIT_TYPE, X_BLOCK_SIZE) vec_t;
#else
 typedef UNIT_TYPE vec_t;
#endif
#if BIAS_TERM
#if SLM_DIV_FACTOR == 1
 vec_t dst = (vec_t)(UNIT_BLOCK_READ(biases, feature_block * FEATURE_SLICE_SIZE));
#else
 vec_t dst;
 if (feature_sub_block == 0) {
 dst = (vec_t)(UNIT_BLOCK_READ(biases, feature_block * FEATURE_SLICE_SIZE));
 } else {
 dst = UNIT_VAL_ZERO;
 }
#endif
#else
 vec_t dst = UNIT_VAL_ZERO;
#endif
#if SLM_DIV_FACTOR > 1
 __local vec_t partial_summ[WORK_GROUP_SIZE];
 for (uint k = feature_sub_block * IC_BLOCKS / SLM_DIV_FACTOR; k < (feature_sub_block + 1) * IC_BLOCKS / SLM_DIV_FACTOR; k++)
 {
#else
 for (uint k = 0; k < IC_BLOCKS; k++)
 {
#endif
 vec_t src = 0;
 if (INPUT_LEFTOVERS && ((k + 1) * FEATURE_SLICE_SIZE >= INPUT0_FEATURE_NUM))
 {
 if (k * FEATURE_SLICE_SIZE + sglid < INPUT0_FEATURE_NUM)
 {
#if X_BLOCK_SIZE > 1
 __attribute__((opencl_unroll_hint(X_BLOCK_SIZE)))
 for (int i = 0; i < X_BLOCK_SIZE; i++)
 {
 const uint xb = (x + i) % INPUT0_SIZE_X;
 const uint yb = y + (x + i) / INPUT0_SIZE_X;
 const uint input_idx = input_offset + k * input_fs_pitch + yb * input_y_pitch + xb * input_x_pitch;
 src[i] = input[input_idx + sglid];
 }
#else
 src = input[input_offset + k * input_fs_pitch + sglid];
#endif
 }
 }
 else
 {
#if PADDED_INPUT
#if X_BLOCK_SIZE > 1
 __attribute__((opencl_unroll_hint(X_BLOCK_SIZE)))
 for (int i = 0; i < X_BLOCK_SIZE; i++)
 {
 const uint xb = (x + i) % INPUT0_SIZE_X;
 const uint yb = y + (x + i) / INPUT0_SIZE_X;
 const uint input_idx = input_offset + k * input_fs_pitch + yb * input_y_pitch + xb * input_x_pitch;
 src[i] = UNIT_BLOCK_READ(input, input_idx);
 }
#else
 src = UNIT_BLOCK_READ(input, input_offset + k * input_fs_pitch);
#endif
#else
#if X_BLOCK_SIZE > 1
 src = UNIT_BLOCK_READ_VEC(input, input_offset + k * input_fs_pitch + input_y * input_y_pitch + input_x * input_x_pitch);
#else
 src = UNIT_BLOCK_READ(input, input_offset + k * input_fs_pitch);
#endif
#endif
 }
 UNIT_TYPE8 wei0 = UNIT_BLOCK_READ8(weights, filter_offset + k * filter_is_pitch);
 UNIT_TYPE8 wei1 = UNIT_BLOCK_READ8(weights, filter_offset + k * filter_is_pitch + 8 * filter_isv_pitch);
 const vec_t src0 = GET_SRC(src, 0);
 const vec_t src1 = GET_SRC(src, 1);
 const vec_t src2 = GET_SRC(src, 2);
 const vec_t src3 = GET_SRC(src, 3);
 const vec_t src4 = GET_SRC(src, 4);
 const vec_t src5 = GET_SRC(src, 5);
 const vec_t src6 = GET_SRC(src, 6);
 const vec_t src7 = GET_SRC(src, 7);
 const vec_t src8 = GET_SRC(src, 8);
 const vec_t src9 = GET_SRC(src, 9);
 const vec_t src10 = GET_SRC(src, 10);
 const vec_t src11 = GET_SRC(src, 11);
 const vec_t src12 = GET_SRC(src, 12);
 const vec_t src13 = GET_SRC(src, 13);
 const vec_t src14 = GET_SRC(src, 14);
 const vec_t src15 = GET_SRC(src, 15);
 dst = mad(wei0.s0, src0, dst);
 dst = mad(wei0.s1, src1, dst);
 dst = mad(wei0.s2, src2, dst);
 dst = mad(wei0.s3, src3, dst);
 dst = mad(wei0.s4, src4, dst);
 dst = mad(wei0.s5, src5, dst);
 dst = mad(wei0.s6, src6, dst);
 dst = mad(wei0.s7, src7, dst);
 dst = mad(wei1.s0, src8, dst);
 dst = mad(wei1.s1, src9, dst);
 dst = mad(wei1.s2, src10, dst);
 dst = mad(wei1.s3, src11, dst);
 dst = mad(wei1.s4, src12, dst);
 dst = mad(wei1.s5, src13, dst);
 dst = mad(wei1.s6, src14, dst);
 dst = mad(wei1.s7, src15, dst);
 }
#if SLM_DIV_FACTOR > 1
 partial_summ[lid1] = dst;
 barrier(CLK_LOCAL_MEM_FENCE);
 if (feature_sub_block == 0) {
 unroll_for(int i = 1; i < SLM_DIV_FACTOR; i++)
 dst += partial_summ[lid1 % feature_per_wg + i * feature_per_wg];
#endif
 dst = ACTIVATION(dst, ACTIVATION_PARAMS);
#if OUTPUT_LEFTOVERS
 if ((feature_block + 1) * FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM)
 {
#if X_BLOCK_SIZE > 1
 for (int i = 0; i < X_BLOCK_SIZE; i++) {
 if (xy * X_BLOCK_SIZE + i >= OUTPUT_SIZE_X * OUTPUT_SIZE_Y || feature_block * FEATURE_SLICE_SIZE + sglid >= OUTPUT_FEATURE_NUM)
 return;
 int xi = (x + i) % OUTPUT_SIZE_X;
 int yi = y + ((x + i) / OUTPUT_SIZE_X);
#if HAS_FUSED_OPS
 FUSED_OPS_SCALAR;
 dst[i] = FUSED_OPS_RESULT_SCALAR;
#endif
 output[output_offset + yi * output_y_pitch + xi * output_x_pitch + sglid] = dst[i];
 }
#else
 if (feature_block * FEATURE_SLICE_SIZE + sglid >= OUTPUT_FEATURE_NUM)
 return;
#if HAS_FUSED_OPS
 FUSED_OPS_SCALAR_B1;
 dst = FUSED_OPS_RESULT_SCALAR_B1;
#endif
 output[output_offset + sglid] = dst;
#endif
 }
 else
#endif
#if X_BLOCK_SIZE > 1
 {
#if !PADDED_OUTPUT && !NON_UNIT_FUSED_OP_SPATIAL
 if (xy * X_BLOCK_SIZE + X_BLOCK_SIZE <= OUTPUT_SIZE_X * OUTPUT_SIZE_Y || (OUTPUT_SIZE_X * OUTPUT_SIZE_Y) % X_BLOCK_SIZE == 0) {
#else
 if (x + X_BLOCK_SIZE <= OUTPUT_SIZE_X || OUTPUT_SIZE_X % X_BLOCK_SIZE == 0) {
#endif
#if HAS_FUSED_OPS
 FUSED_OPS_VEC;
 dst = FUSED_OPS_RESULT_VEC;
#endif
 UNIT_BLOCK_WRITE_VEC(output, output_offset + y * output_y_pitch + x * output_x_pitch, dst);
 } else {
 for (int i = 0; i < X_BLOCK_SIZE; i++) {
 if (xy * X_BLOCK_SIZE + i >= OUTPUT_SIZE_X * OUTPUT_SIZE_Y)
 return;
 int xi = (x + i) % OUTPUT_SIZE_X;
 int yi = y + ((x + i) / OUTPUT_SIZE_X);
#if HAS_FUSED_OPS
 FUSED_OPS_SCALAR;
 dst[i] = FUSED_OPS_RESULT_SCALAR;
#endif
 UNIT_BLOCK_WRITE(output, output_offset + yi * output_y_pitch + xi * output_x_pitch, dst[i]);
 }
 }
 }
#else
 {
#if HAS_FUSED_OPS
 FUSED_OPS_SCALAR_B1;
 dst = FUSED_OPS_RESULT_SCALAR_B1;
#endif
 UNIT_BLOCK_WRITE(output, output_offset, dst);
 }
#endif
#if SLM_DIV_FACTOR > 1
 }
#endif
}
#undef GET_SRC
#undef FEATURE_SLICE_SIZE
#undef UNIT_BLOCK_READ_VEC
#undef UNIT_BLOCK_WRITE_VEC
#ifdef FEATURE_SLICE_SIZE
#undef FEATURE_SLICE_SIZE
#endif
#ifdef UNIT_BLOCK_RW_TYPE
#undef UNIT_BLOCK_RW_TYPE
#endif
#ifdef UNIT_BLOCK_READ_FUNC
#undef UNIT_BLOCK_READ_FUNC
#endif
#ifdef UNIT_BLOCK_WRITE_FUNC
#undef UNIT_BLOCK_WRITE_FUNC
#endif
#ifdef UNIT_BLOCK_RW_TYPE
#undef UNIT_BLOCK_RW_TYPE
#endif
#ifdef UNIT_BLOCK_READ_FUNC
#undef UNIT_BLOCK_READ_FUNC
#endif
#ifdef UNIT_BLOCK_WRITE_FUNC
#undef UNIT_BLOCK_WRITE_FUNC
#endif
#ifdef UNIT_TYPE2
#undef UNIT_TYPE2
#endif
#ifdef UNIT_TYPE4
#undef UNIT_TYPE4
#endif
#ifdef UNIT_TYPE8
#undef UNIT_TYPE8
#endif
#ifdef UNIT_BLOCK_RW_TYPE2
#undef UNIT_BLOCK_RW_TYPE2
#endif
#ifdef UNIT_BLOCK_RW_TYPE4
#undef UNIT_BLOCK_RW_TYPE4
#endif
#ifdef UNIT_BLOCK_RW_TYPE8
#undef UNIT_BLOCK_RW_TYPE8
#endif
#ifdef UNIT_BLOCK_READ_FUNC2
#undef UNIT_BLOCK_READ_FUNC2
#endif
#ifdef UNIT_BLOCK_READ_FUNC4
#undef UNIT_BLOCK_READ_FUNC4
#endif
#ifdef UNIT_BLOCK_READ_FUNC8
#undef UNIT_BLOCK_READ_FUNC8
#endif
#ifdef UNIT_BLOCK_WRITE_FUNC2
#undef UNIT_BLOCK_WRITE_FUNC2
#endif
#ifdef UNIT_BLOCK_WRITE_FUNC4
#undef UNIT_BLOCK_WRITE_FUNC4
#endif
#ifdef UNIT_BLOCK_WRITE_FUNC8
#undef UNIT_BLOCK_WRITE_FUNC8
#endif
#ifdef UNIT_BLOCK_READ
#undef UNIT_BLOCK_READ
#endif
#ifdef UNIT_BLOCK_READ2
#undef UNIT_BLOCK_READ2
#endif
#ifdef UNIT_BLOCK_READ4
#undef UNIT_BLOCK_READ4
#endif
#ifdef UNIT_BLOCK_READ8
#undef UNIT_BLOCK_READ8
#endif
#ifdef UNIT_BLOCK_WRITE
#undef UNIT_BLOCK_WRITE
#endif
#ifdef UNIT_BLOCK_WRITE2
#undef UNIT_BLOCK_WRITE2
#endif
#ifdef UNIT_BLOCK_WRITE4
#undef UNIT_BLOCK_WRITE4
#endif
#ifdef UNIT_BLOCK_WRITE8
#undef UNIT_BLOCK_WRITE8
#endif
#undef KERNEL
#undef KERNEL_ID
#undef FUNC
#undef FUNC_CALL
#undef CONST_ARRAY_DECL
#undef CONST_ARRAY_REF
#ifdef FP64_SUPPORTED
#undef FP64_SUPPORTED
#endif
#ifdef FP16_SUPPORTED
#undef FP16_SUPPORTED
#endif
#ifdef FP16_UNIT_USED
#undef FP16_UNIT_USED
#endif
#ifdef INT8_UNIT_USED
#undef INT8_UNIT_USED
#endif
#ifdef INT32_UNIT_USED
#undef INT32_UNIT_USED
#endif
#ifdef INT64_UNIT_USED
#undef INT64_UNIT_USED
#endif
#ifdef UINT8_UNIT_USED
#undef UINT8_UNIT_USED
#endif
#ifdef UINT32_UNIT_USED
#undef UINT32_UNIT_USED
#endif
#ifdef UNIT_TYPE
#undef UNIT_TYPE
#endif
#ifdef UNIT_VAL_MAX
#undef UNIT_VAL_MAX
#endif
#ifdef UNIT_VAL_MIN
#undef UNIT_VAL_MIN
#endif
#ifdef UNIT_VAL_ONE
#undef UNIT_VAL_ONE
#endif
#ifdef UNIT_VAL_ZERO
#undef UNIT_VAL_ZERO
#endif
#ifdef TO_UNIT_TYPE
#undef TO_UNIT_TYPE
#endif
#ifdef TO_UNIT_TYPE_SAT
#undef TO_UNIT_TYPE_SAT
#endif
#ifdef AS_UNIT_TYPE
#undef AS_UNIT_TYPE
#endif
#ifdef UNIT_MAX_FUNC
#undef UNIT_MAX_FUNC
#endif
#ifdef UNIT_MIN_FUNC
#undef UNIT_MIN_FUNC
#endif
#ifdef UNIT_ABS_FUNC
#undef UNIT_ABS_FUNC
#endif
#ifdef UNIT_TYPE_SIZE
#undef UNIT_TYPE_SIZE
#endif
#ifdef UNIT_IS_FP
#undef UNIT_IS_FP
#endif
#ifdef NL_M
#undef NL_M
#endif
#ifdef NL_N
#undef NL_N
#endif
#ifdef ACTIVATION_FUNC_TYPE
#undef ACTIVATION_FUNC_TYPE
#endif
#ifdef ACTIVATION_FUNC_VAL_MAX
#undef ACTIVATION_FUNC_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_VAL_MIN
#undef ACTIVATION_FUNC_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_VAL_ONE
#undef ACTIVATION_FUNC_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_VAL_ZERO
#undef ACTIVATION_FUNC_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_TYPE
#undef TO_ACTIVATION_FUNC_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_TYPE_SAT
#undef TO_ACTIVATION_FUNC_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_TYPE
#undef AS_ACTIVATION_FUNC_TYPE
#endif
#ifdef ACTIVATION_FUNC_MAX_FUNC
#undef ACTIVATION_FUNC_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_MIN_FUNC
#undef ACTIVATION_FUNC_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_ABS_FUNC
#undef ACTIVATION_FUNC_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_TYPE_SIZE
#undef ACTIVATION_FUNC_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_IS_FP
#undef ACTIVATION_FUNC_IS_FP
#endif
#ifdef ACTIVATION_PARAMS
#undef ACTIVATION_PARAMS
#endif
#ifdef ACTIVATION_FUNC
#undef ACTIVATION_FUNC
#endif
#ifdef ACTIVATION
#undef ACTIVATION
#endif
#ifdef INPUT0_SIZE_X
#undef INPUT0_SIZE_X
#endif
#ifdef INPUT0_SIZE_Y
#undef INPUT0_SIZE_Y
#endif
#ifdef INPUT0_SIZE_Z
#undef INPUT0_SIZE_Z
#endif
#ifdef INPUT0_SIZE_W
#undef INPUT0_SIZE_W
#endif
#ifdef INPUT0_SIZE_U
#undef INPUT0_SIZE_U
#endif
#ifdef INPUT0_SIZE_V
#undef INPUT0_SIZE_V
#endif
#ifdef INPUT0_FEATURE_NUM
#undef INPUT0_FEATURE_NUM
#endif
#ifdef INPUT0_BATCH_NUM
#undef INPUT0_BATCH_NUM
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_X
#undef INPUT0_PAD_BEFORE_SIZE_X
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_Y
#undef INPUT0_PAD_BEFORE_SIZE_Y
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_Z
#undef INPUT0_PAD_BEFORE_SIZE_Z
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_W
#undef INPUT0_PAD_BEFORE_SIZE_W
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_U
#undef INPUT0_PAD_BEFORE_SIZE_U
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_V
#undef INPUT0_PAD_BEFORE_SIZE_V
#endif
#ifdef INPUT0_PAD_BEFORE_FEATURE_NUM
#undef INPUT0_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef INPUT0_PAD_BEFORE_BATCH_NUM
#undef INPUT0_PAD_BEFORE_BATCH_NUM
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_X
#undef INPUT0_PAD_AFTER_SIZE_X
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_Y
#undef INPUT0_PAD_AFTER_SIZE_Y
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_Z
#undef INPUT0_PAD_AFTER_SIZE_Z
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_W
#undef INPUT0_PAD_AFTER_SIZE_W
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_U
#undef INPUT0_PAD_AFTER_SIZE_U
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_V
#undef INPUT0_PAD_AFTER_SIZE_V
#endif
#ifdef INPUT0_PAD_AFTER_FEATURE_NUM
#undef INPUT0_PAD_AFTER_FEATURE_NUM
#endif
#ifdef INPUT0_PAD_AFTER_BATCH_NUM
#undef INPUT0_PAD_AFTER_BATCH_NUM
#endif
#ifdef INPUT0_LENGTH
#undef INPUT0_LENGTH
#endif
#ifdef INPUT0_X_PITCH
#undef INPUT0_X_PITCH
#endif
#ifdef INPUT0_Y_PITCH
#undef INPUT0_Y_PITCH
#endif
#ifdef INPUT0_Z_PITCH
#undef INPUT0_Z_PITCH
#endif
#ifdef INPUT0_W_PITCH
#undef INPUT0_W_PITCH
#endif
#ifdef INPUT0_U_PITCH
#undef INPUT0_U_PITCH
#endif
#ifdef INPUT0_V_PITCH
#undef INPUT0_V_PITCH
#endif
#ifdef INPUT0_FEATURE_PITCH
#undef INPUT0_FEATURE_PITCH
#endif
#ifdef INPUT0_BATCH_PITCH
#undef INPUT0_BATCH_PITCH
#endif
#ifdef INPUT0_GET_INDEX_SAFE
#undef INPUT0_GET_INDEX_SAFE
#endif
#ifdef INPUT0_GET_INDEX
#undef INPUT0_GET_INDEX
#endif
#ifdef INPUT0_GET_INDEX_RAW
#undef INPUT0_GET_INDEX_RAW
#endif
#ifdef INPUT0_VIEW_OFFSET
#undef INPUT0_VIEW_OFFSET
#endif
#ifdef INPUT0_DIMS
#undef INPUT0_DIMS
#endif
#ifdef INPUT0_SIMPLE
#undef INPUT0_SIMPLE
#endif
#ifdef INPUT0_GROUPED
#undef INPUT0_GROUPED
#endif
#ifdef INPUT0_LAYOUT_B_FS_YX_FSV16
#undef INPUT0_LAYOUT_B_FS_YX_FSV16
#endif
#ifdef INPUT0_TYPE
#undef INPUT0_TYPE
#endif
#ifdef INPUT0_VAL_MAX
#undef INPUT0_VAL_MAX
#endif
#ifdef INPUT0_VAL_MIN
#undef INPUT0_VAL_MIN
#endif
#ifdef INPUT0_VAL_ONE
#undef INPUT0_VAL_ONE
#endif
#ifdef INPUT0_VAL_ZERO
#undef INPUT0_VAL_ZERO
#endif
#ifdef TO_INPUT0_TYPE
#undef TO_INPUT0_TYPE
#endif
#ifdef TO_INPUT0_TYPE_SAT
#undef TO_INPUT0_TYPE_SAT
#endif
#ifdef AS_INPUT0_TYPE
#undef AS_INPUT0_TYPE
#endif
#ifdef INPUT0_MAX_FUNC
#undef INPUT0_MAX_FUNC
#endif
#ifdef INPUT0_MIN_FUNC
#undef INPUT0_MIN_FUNC
#endif
#ifdef INPUT0_ABS_FUNC
#undef INPUT0_ABS_FUNC
#endif
#ifdef INPUT0_TYPE_SIZE
#undef INPUT0_TYPE_SIZE
#endif
#ifdef INPUT0_IS_FP
#undef INPUT0_IS_FP
#endif
#ifdef INPUT0_OFFSET
#undef INPUT0_OFFSET
#endif
#ifdef INPUT0_SIZES_DATA
#undef INPUT0_SIZES_DATA
#endif
#ifdef INPUT0_PITCHES
#undef INPUT0_PITCHES
#endif
#ifdef INPUT0_PAD_BEFORE
#undef INPUT0_PAD_BEFORE
#endif
#ifdef INPUT0_PAD_AFTER
#undef INPUT0_PAD_AFTER
#endif
#ifdef OUTPUT_SIZE_X
#undef OUTPUT_SIZE_X
#endif
#ifdef OUTPUT_SIZE_Y
#undef OUTPUT_SIZE_Y
#endif
#ifdef OUTPUT_SIZE_Z
#undef OUTPUT_SIZE_Z
#endif
#ifdef OUTPUT_SIZE_W
#undef OUTPUT_SIZE_W
#endif
#ifdef OUTPUT_SIZE_U
#undef OUTPUT_SIZE_U
#endif
#ifdef OUTPUT_SIZE_V
#undef OUTPUT_SIZE_V
#endif
#ifdef OUTPUT_FEATURE_NUM
#undef OUTPUT_FEATURE_NUM
#endif
#ifdef OUTPUT_BATCH_NUM
#undef OUTPUT_BATCH_NUM
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_X
#undef OUTPUT_PAD_BEFORE_SIZE_X
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_Y
#undef OUTPUT_PAD_BEFORE_SIZE_Y
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_Z
#undef OUTPUT_PAD_BEFORE_SIZE_Z
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_W
#undef OUTPUT_PAD_BEFORE_SIZE_W
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_U
#undef OUTPUT_PAD_BEFORE_SIZE_U
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_V
#undef OUTPUT_PAD_BEFORE_SIZE_V
#endif
#ifdef OUTPUT_PAD_BEFORE_FEATURE_NUM
#undef OUTPUT_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef OUTPUT_PAD_BEFORE_BATCH_NUM
#undef OUTPUT_PAD_BEFORE_BATCH_NUM
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_X
#undef OUTPUT_PAD_AFTER_SIZE_X
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_Y
#undef OUTPUT_PAD_AFTER_SIZE_Y
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_Z
#undef OUTPUT_PAD_AFTER_SIZE_Z
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_W
#undef OUTPUT_PAD_AFTER_SIZE_W
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_U
#undef OUTPUT_PAD_AFTER_SIZE_U
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_V
#undef OUTPUT_PAD_AFTER_SIZE_V
#endif
#ifdef OUTPUT_PAD_AFTER_FEATURE_NUM
#undef OUTPUT_PAD_AFTER_FEATURE_NUM
#endif
#ifdef OUTPUT_PAD_AFTER_BATCH_NUM
#undef OUTPUT_PAD_AFTER_BATCH_NUM
#endif
#ifdef OUTPUT_LENGTH
#undef OUTPUT_LENGTH
#endif
#ifdef OUTPUT_X_PITCH
#undef OUTPUT_X_PITCH
#endif
#ifdef OUTPUT_Y_PITCH
#undef OUTPUT_Y_PITCH
#endif
#ifdef OUTPUT_Z_PITCH
#undef OUTPUT_Z_PITCH
#endif
#ifdef OUTPUT_W_PITCH
#undef OUTPUT_W_PITCH
#endif
#ifdef OUTPUT_U_PITCH
#undef OUTPUT_U_PITCH
#endif
#ifdef OUTPUT_V_PITCH
#undef OUTPUT_V_PITCH
#endif
#ifdef OUTPUT_FEATURE_PITCH
#undef OUTPUT_FEATURE_PITCH
#endif
#ifdef OUTPUT_BATCH_PITCH
#undef OUTPUT_BATCH_PITCH
#endif
#ifdef OUTPUT_GET_INDEX_SAFE
#undef OUTPUT_GET_INDEX_SAFE
#endif
#ifdef OUTPUT_GET_INDEX
#undef OUTPUT_GET_INDEX
#endif
#ifdef OUTPUT_GET_INDEX_RAW
#undef OUTPUT_GET_INDEX_RAW
#endif
#ifdef OUTPUT_VIEW_OFFSET
#undef OUTPUT_VIEW_OFFSET
#endif
#ifdef OUTPUT_DIMS
#undef OUTPUT_DIMS
#endif
#ifdef OUTPUT_SIMPLE
#undef OUTPUT_SIMPLE
#endif
#ifdef OUTPUT_GROUPED
#undef OUTPUT_GROUPED
#endif
#ifdef OUTPUT_LAYOUT_B_FS_YX_FSV16
#undef OUTPUT_LAYOUT_B_FS_YX_FSV16
#endif
#ifdef OUTPUT_TYPE
#undef OUTPUT_TYPE
#endif
#ifdef OUTPUT_VAL_MAX
#undef OUTPUT_VAL_MAX
#endif
#ifdef OUTPUT_VAL_MIN
#undef OUTPUT_VAL_MIN
#endif
#ifdef OUTPUT_VAL_ONE
#undef OUTPUT_VAL_ONE
#endif
#ifdef OUTPUT_VAL_ZERO
#undef OUTPUT_VAL_ZERO
#endif
#ifdef TO_OUTPUT_TYPE
#undef TO_OUTPUT_TYPE
#endif
#ifdef TO_OUTPUT_TYPE_SAT
#undef TO_OUTPUT_TYPE_SAT
#endif
#ifdef AS_OUTPUT_TYPE
#undef AS_OUTPUT_TYPE
#endif
#ifdef OUTPUT_MAX_FUNC
#undef OUTPUT_MAX_FUNC
#endif
#ifdef OUTPUT_MIN_FUNC
#undef OUTPUT_MIN_FUNC
#endif
#ifdef OUTPUT_ABS_FUNC
#undef OUTPUT_ABS_FUNC
#endif
#ifdef OUTPUT_TYPE_SIZE
#undef OUTPUT_TYPE_SIZE
#endif
#ifdef OUTPUT_IS_FP
#undef OUTPUT_IS_FP
#endif
#ifdef OUTPUT_OFFSET
#undef OUTPUT_OFFSET
#endif
#ifdef OUTPUT_SIZES_DATA
#undef OUTPUT_SIZES_DATA
#endif
#ifdef OUTPUT_PITCHES
#undef OUTPUT_PITCHES
#endif
#ifdef OUTPUT_PAD_BEFORE
#undef OUTPUT_PAD_BEFORE
#endif
#ifdef OUTPUT_PAD_AFTER
#undef OUTPUT_PAD_AFTER
#endif
#ifdef OPTIONAL_SHAPE_INFO_ARG
#undef OPTIONAL_SHAPE_INFO_ARG
#endif
#ifdef OPTIONAL_SHAPE_INFO_TENSOR
#undef OPTIONAL_SHAPE_INFO_TENSOR
#endif
#ifdef FILTER_SIZE_X
#undef FILTER_SIZE_X
#endif
#ifdef FILTER_SIZE_Y
#undef FILTER_SIZE_Y
#endif
#ifdef FILTER_SIZE_Z
#undef FILTER_SIZE_Z
#endif
#ifdef FILTER_IFM_NUM
#undef FILTER_IFM_NUM
#endif
#ifdef FILTER_OFM_NUM
#undef FILTER_OFM_NUM
#endif
#ifdef FILTER_GROUPS_NUM
#undef FILTER_GROUPS_NUM
#endif
#ifdef FILTER_X_PITCH
#undef FILTER_X_PITCH
#endif
#ifdef FILTER_Y_PITCH
#undef FILTER_Y_PITCH
#endif
#ifdef FILTER_Z_PITCH
#undef FILTER_Z_PITCH
#endif
#ifdef FILTER_IFM_PITCH
#undef FILTER_IFM_PITCH
#endif
#ifdef FILTER_OFM_PITCH
#undef FILTER_OFM_PITCH
#endif
#ifdef FILTER_GROUPS_PITCH
#undef FILTER_GROUPS_PITCH
#endif
#ifdef FILTER_LENGTH
#undef FILTER_LENGTH
#endif
#ifdef FILTER_VIEW_OFFSET
#undef FILTER_VIEW_OFFSET
#endif
#ifdef FILTER_DIMS
#undef FILTER_DIMS
#endif
#ifdef FILTER_SIMPLE
#undef FILTER_SIMPLE
#endif
#ifdef FILTER_GROUPED
#undef FILTER_GROUPED
#endif
#ifdef FILTER_LAYOUT_OS_IS_YX_ISV16_OSV16
#undef FILTER_LAYOUT_OS_IS_YX_ISV16_OSV16
#endif
#ifdef FILTER_TYPE
#undef FILTER_TYPE
#endif
#ifdef FILTER_VAL_MAX
#undef FILTER_VAL_MAX
#endif
#ifdef FILTER_VAL_MIN
#undef FILTER_VAL_MIN
#endif
#ifdef FILTER_VAL_ONE
#undef FILTER_VAL_ONE
#endif
#ifdef FILTER_VAL_ZERO
#undef FILTER_VAL_ZERO
#endif
#ifdef TO_FILTER_TYPE
#undef TO_FILTER_TYPE
#endif
#ifdef TO_FILTER_TYPE_SAT
#undef TO_FILTER_TYPE_SAT
#endif
#ifdef AS_FILTER_TYPE
#undef AS_FILTER_TYPE
#endif
#ifdef FILTER_MAX_FUNC
#undef FILTER_MAX_FUNC
#endif
#ifdef FILTER_MIN_FUNC
#undef FILTER_MIN_FUNC
#endif
#ifdef FILTER_ABS_FUNC
#undef FILTER_ABS_FUNC
#endif
#ifdef FILTER_TYPE_SIZE
#undef FILTER_TYPE_SIZE
#endif
#ifdef FILTER_IS_FP
#undef FILTER_IS_FP
#endif
#ifdef FILTER_OFFSET
#undef FILTER_OFFSET
#endif
#ifdef FILTER_SIZES_DATA
#undef FILTER_SIZES_DATA
#endif
#ifdef FILTER_PITCHES
#undef FILTER_PITCHES
#endif
#ifdef FILTER_PAD_BEFORE
#undef FILTER_PAD_BEFORE
#endif
#ifdef FILTER_PAD_AFTER
#undef FILTER_PAD_AFTER
#endif
#ifdef BIAS_TERM
#undef BIAS_TERM
#endif
#ifdef BIAS_SIZE_X
#undef BIAS_SIZE_X
#endif
#ifdef BIAS_SIZE_Y
#undef BIAS_SIZE_Y
#endif
#ifdef BIAS_SIZE_Z
#undef BIAS_SIZE_Z
#endif
#ifdef BIAS_SIZE_W
#undef BIAS_SIZE_W
#endif
#ifdef BIAS_SIZE_U
#undef BIAS_SIZE_U
#endif
#ifdef BIAS_SIZE_V
#undef BIAS_SIZE_V
#endif
#ifdef BIAS_FEATURE_NUM
#undef BIAS_FEATURE_NUM
#endif
#ifdef BIAS_BATCH_NUM
#undef BIAS_BATCH_NUM
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_X
#undef BIAS_PAD_BEFORE_SIZE_X
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_Y
#undef BIAS_PAD_BEFORE_SIZE_Y
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_Z
#undef BIAS_PAD_BEFORE_SIZE_Z
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_W
#undef BIAS_PAD_BEFORE_SIZE_W
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_U
#undef BIAS_PAD_BEFORE_SIZE_U
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_V
#undef BIAS_PAD_BEFORE_SIZE_V
#endif
#ifdef BIAS_PAD_BEFORE_FEATURE_NUM
#undef BIAS_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef BIAS_PAD_BEFORE_BATCH_NUM
#undef BIAS_PAD_BEFORE_BATCH_NUM
#endif
#ifdef BIAS_PAD_AFTER_SIZE_X
#undef BIAS_PAD_AFTER_SIZE_X
#endif
#ifdef BIAS_PAD_AFTER_SIZE_Y
#undef BIAS_PAD_AFTER_SIZE_Y
#endif
#ifdef BIAS_PAD_AFTER_SIZE_Z
#undef BIAS_PAD_AFTER_SIZE_Z
#endif
#ifdef BIAS_PAD_AFTER_SIZE_W
#undef BIAS_PAD_AFTER_SIZE_W
#endif
#ifdef BIAS_PAD_AFTER_SIZE_U
#undef BIAS_PAD_AFTER_SIZE_U
#endif
#ifdef BIAS_PAD_AFTER_SIZE_V
#undef BIAS_PAD_AFTER_SIZE_V
#endif
#ifdef BIAS_PAD_AFTER_FEATURE_NUM
#undef BIAS_PAD_AFTER_FEATURE_NUM
#endif
#ifdef BIAS_PAD_AFTER_BATCH_NUM
#undef BIAS_PAD_AFTER_BATCH_NUM
#endif
#ifdef BIAS_LENGTH
#undef BIAS_LENGTH
#endif
#ifdef BIAS_X_PITCH
#undef BIAS_X_PITCH
#endif
#ifdef BIAS_Y_PITCH
#undef BIAS_Y_PITCH
#endif
#ifdef BIAS_Z_PITCH
#undef BIAS_Z_PITCH
#endif
#ifdef BIAS_W_PITCH
#undef BIAS_W_PITCH
#endif
#ifdef BIAS_U_PITCH
#undef BIAS_U_PITCH
#endif
#ifdef BIAS_V_PITCH
#undef BIAS_V_PITCH
#endif
#ifdef BIAS_FEATURE_PITCH
#undef BIAS_FEATURE_PITCH
#endif
#ifdef BIAS_BATCH_PITCH
#undef BIAS_BATCH_PITCH
#endif
#ifdef BIAS_GET_INDEX_SAFE
#undef BIAS_GET_INDEX_SAFE
#endif
#ifdef BIAS_GET_INDEX
#undef BIAS_GET_INDEX
#endif
#ifdef BIAS_GET_INDEX_RAW
#undef BIAS_GET_INDEX_RAW
#endif
#ifdef BIAS_VIEW_OFFSET
#undef BIAS_VIEW_OFFSET
#endif
#ifdef BIAS_DIMS
#undef BIAS_DIMS
#endif
#ifdef BIAS_SIMPLE
#undef BIAS_SIMPLE
#endif
#ifdef BIAS_GROUPED
#undef BIAS_GROUPED
#endif
#ifdef BIAS_LAYOUT_BF
#undef BIAS_LAYOUT_BF
#endif
#ifdef BIAS_TYPE
#undef BIAS_TYPE
#endif
#ifdef BIAS_VAL_MAX
#undef BIAS_VAL_MAX
#endif
#ifdef BIAS_VAL_MIN
#undef BIAS_VAL_MIN
#endif
#ifdef BIAS_VAL_ONE
#undef BIAS_VAL_ONE
#endif
#ifdef BIAS_VAL_ZERO
#undef BIAS_VAL_ZERO
#endif
#ifdef TO_BIAS_TYPE
#undef TO_BIAS_TYPE
#endif
#ifdef TO_BIAS_TYPE_SAT
#undef TO_BIAS_TYPE_SAT
#endif
#ifdef AS_BIAS_TYPE
#undef AS_BIAS_TYPE
#endif
#ifdef BIAS_MAX_FUNC
#undef BIAS_MAX_FUNC
#endif
#ifdef BIAS_MIN_FUNC
#undef BIAS_MIN_FUNC
#endif
#ifdef BIAS_ABS_FUNC
#undef BIAS_ABS_FUNC
#endif
#ifdef BIAS_TYPE_SIZE
#undef BIAS_TYPE_SIZE
#endif
#ifdef BIAS_IS_FP
#undef BIAS_IS_FP
#endif
#ifdef BIAS_OFFSET
#undef BIAS_OFFSET
#endif
#ifdef BIAS_SIZES_DATA
#undef BIAS_SIZES_DATA
#endif
#ifdef BIAS_PITCHES
#undef BIAS_PITCHES
#endif
#ifdef BIAS_PAD_BEFORE
#undef BIAS_PAD_BEFORE
#endif
#ifdef BIAS_PAD_AFTER
#undef BIAS_PAD_AFTER
#endif
#ifdef BIAS_PER_OUTPUT
#undef BIAS_PER_OUTPUT
#endif
#ifdef BIAS_PER_OFM
#undef BIAS_PER_OFM
#endif
#ifdef STRIDE_SIZE_X
#undef STRIDE_SIZE_X
#endif
#ifdef STRIDE_SIZE_Y
#undef STRIDE_SIZE_Y
#endif
#ifdef STRIDE_SIZE_Z
#undef STRIDE_SIZE_Z
#endif
#ifdef PADDING_SIZE_X
#undef PADDING_SIZE_X
#endif
#ifdef PADDING_SIZE_Y
#undef PADDING_SIZE_Y
#endif
#ifdef PADDING_SIZE_Z
#undef PADDING_SIZE_Z
#endif
#ifdef DILATION_SIZE_X
#undef DILATION_SIZE_X
#endif
#ifdef DILATION_SIZE_Y
#undef DILATION_SIZE_Y
#endif
#ifdef DILATION_SIZE_Z
#undef DILATION_SIZE_Z
#endif
#ifdef FILTER_ARRAY_NUM
#undef FILTER_ARRAY_NUM
#endif
#ifdef INPUT0_OFFSET_WITH_PADDING
#undef INPUT0_OFFSET_WITH_PADDING
#endif
#ifdef GROUPED
#undef GROUPED
#endif
#ifdef SKIP_BATCH
#undef SKIP_BATCH
#endif
#ifdef FUSED_OP0_LOAD_VEC
#undef FUSED_OP0_LOAD_VEC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_MAX
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_MIN
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_ONE
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_ZERO
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE
#undef TO_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE_SAT
#undef TO_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE
#undef AS_ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_MAX_FUNC
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_MIN_FUNC
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_ABS_FUNC
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE_SIZE
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC_IS_FP
#undef ACTIVATION_FUNC_FUSED_OP0_VEC_IS_FP
#endif
#ifdef ACTIVATION_PARAMS_FUSED_OP0_VEC
#undef ACTIVATION_PARAMS_FUSED_OP0_VEC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_VEC
#undef ACTIVATION_FUNC_FUSED_OP0_VEC
#endif
#ifdef ACTIVATION_FUSED_OP0_VEC
#undef ACTIVATION_FUSED_OP0_VEC
#endif
#ifdef FUSED_OP0_ACTION_VEC
#undef FUSED_OP0_ACTION_VEC
#endif
#ifdef FUSED_OPS_VEC
#undef FUSED_OPS_VEC
#endif
#ifdef FUSED_OPS_PRELOAD_VEC
#undef FUSED_OPS_PRELOAD_VEC
#endif
#ifdef FUSED_OPS_CALC_VEC
#undef FUSED_OPS_CALC_VEC
#endif
#ifdef FUSED_OPS_RESULT_VEC
#undef FUSED_OPS_RESULT_VEC
#endif
#ifdef FUSED_OPS_CAN_USE_PRELOAD_VEC
#undef FUSED_OPS_CAN_USE_PRELOAD_VEC
#endif
#ifdef FUSED_OP0_LOAD_SCALAR
#undef FUSED_OP0_LOAD_SCALAR
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_MAX
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_MIN
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_ONE
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_ZERO
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE
#undef TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE_SAT
#undef TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE
#undef AS_ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_MAX_FUNC
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_MIN_FUNC
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_ABS_FUNC
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE_SIZE
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_IS_FP
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_IS_FP
#endif
#ifdef ACTIVATION_PARAMS_FUSED_OP0_SCALAR
#undef ACTIVATION_PARAMS_FUSED_OP0_SCALAR
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR
#endif
#ifdef ACTIVATION_FUSED_OP0_SCALAR
#undef ACTIVATION_FUSED_OP0_SCALAR
#endif
#ifdef FUSED_OP0_ACTION_SCALAR
#undef FUSED_OP0_ACTION_SCALAR
#endif
#ifdef FUSED_OPS_SCALAR
#undef FUSED_OPS_SCALAR
#endif
#ifdef FUSED_OPS_PRELOAD_SCALAR
#undef FUSED_OPS_PRELOAD_SCALAR
#endif
#ifdef FUSED_OPS_CALC_SCALAR
#undef FUSED_OPS_CALC_SCALAR
#endif
#ifdef FUSED_OPS_RESULT_SCALAR
#undef FUSED_OPS_RESULT_SCALAR
#endif
#ifdef FUSED_OPS_CAN_USE_PRELOAD_SCALAR
#undef FUSED_OPS_CAN_USE_PRELOAD_SCALAR
#endif
#ifdef FUSED_OP0_LOAD_SCALAR_B1
#undef FUSED_OP0_LOAD_SCALAR_B1
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_TYPE
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_VAL_MAX
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_VAL_MIN
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_VAL_ONE
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_VAL_ZERO
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_TYPE
#undef TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_TYPE_SAT
#undef TO_ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_TYPE
#undef AS_ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_MAX_FUNC
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_MIN_FUNC
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_ABS_FUNC
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_TYPE_SIZE
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_IS_FP
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1_IS_FP
#endif
#ifdef ACTIVATION_PARAMS_FUSED_OP0_SCALAR_B1
#undef ACTIVATION_PARAMS_FUSED_OP0_SCALAR_B1
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1
#undef ACTIVATION_FUNC_FUSED_OP0_SCALAR_B1
#endif
#ifdef ACTIVATION_FUSED_OP0_SCALAR_B1
#undef ACTIVATION_FUSED_OP0_SCALAR_B1
#endif
#ifdef FUSED_OP0_ACTION_SCALAR_B1
#undef FUSED_OP0_ACTION_SCALAR_B1
#endif
#ifdef FUSED_OPS_SCALAR_B1
#undef FUSED_OPS_SCALAR_B1
#endif
#ifdef FUSED_OPS_PRELOAD_SCALAR_B1
#undef FUSED_OPS_PRELOAD_SCALAR_B1
#endif
#ifdef FUSED_OPS_CALC_SCALAR_B1
#undef FUSED_OPS_CALC_SCALAR_B1
#endif
#ifdef FUSED_OPS_RESULT_SCALAR_B1
#undef FUSED_OPS_RESULT_SCALAR_B1
#endif
#ifdef FUSED_OPS_CAN_USE_PRELOAD_SCALAR_B1
#undef FUSED_OPS_CAN_USE_PRELOAD_SCALAR_B1
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_X
#undef FUSED_OP_0_OUTPUT_SIZE_X
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_Y
#undef FUSED_OP_0_OUTPUT_SIZE_Y
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_Z
#undef FUSED_OP_0_OUTPUT_SIZE_Z
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_W
#undef FUSED_OP_0_OUTPUT_SIZE_W
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_U
#undef FUSED_OP_0_OUTPUT_SIZE_U
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_V
#undef FUSED_OP_0_OUTPUT_SIZE_V
#endif
#ifdef FUSED_OP_0_OUTPUT_FEATURE_NUM
#undef FUSED_OP_0_OUTPUT_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_BATCH_NUM
#undef FUSED_OP_0_OUTPUT_BATCH_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_X
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_X
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Y
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Y
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Z
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Z
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_W
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_W
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_U
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_U
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_V
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_V
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_FEATURE_NUM
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_BATCH_NUM
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_BATCH_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_X
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_X
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Y
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Y
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Z
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Z
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_W
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_W
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_U
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_U
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_V
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_V
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_FEATURE_NUM
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_BATCH_NUM
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_BATCH_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_LENGTH
#undef FUSED_OP_0_OUTPUT_LENGTH
#endif
#ifdef FUSED_OP_0_OUTPUT_X_PITCH
#undef FUSED_OP_0_OUTPUT_X_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_Y_PITCH
#undef FUSED_OP_0_OUTPUT_Y_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_Z_PITCH
#undef FUSED_OP_0_OUTPUT_Z_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_W_PITCH
#undef FUSED_OP_0_OUTPUT_W_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_U_PITCH
#undef FUSED_OP_0_OUTPUT_U_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_V_PITCH
#undef FUSED_OP_0_OUTPUT_V_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_FEATURE_PITCH
#undef FUSED_OP_0_OUTPUT_FEATURE_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_BATCH_PITCH
#undef FUSED_OP_0_OUTPUT_BATCH_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_GET_INDEX_SAFE
#undef FUSED_OP_0_OUTPUT_GET_INDEX_SAFE
#endif
#ifdef FUSED_OP_0_OUTPUT_GET_INDEX
#undef FUSED_OP_0_OUTPUT_GET_INDEX
#endif
#ifdef FUSED_OP_0_OUTPUT_GET_INDEX_RAW
#undef FUSED_OP_0_OUTPUT_GET_INDEX_RAW
#endif
#ifdef FUSED_OP_0_OUTPUT_VIEW_OFFSET
#undef FUSED_OP_0_OUTPUT_VIEW_OFFSET
#endif
#ifdef FUSED_OP_0_OUTPUT_DIMS
#undef FUSED_OP_0_OUTPUT_DIMS
#endif
#ifdef FUSED_OP_0_OUTPUT_SIMPLE
#undef FUSED_OP_0_OUTPUT_SIMPLE
#endif
#ifdef FUSED_OP_0_OUTPUT_GROUPED
#undef FUSED_OP_0_OUTPUT_GROUPED
#endif
#ifdef FUSED_OP_0_OUTPUT_LAYOUT_BFYX
#undef FUSED_OP_0_OUTPUT_LAYOUT_BFYX
#endif
#ifdef FUSED_OP_0_OUTPUT_TYPE
#undef FUSED_OP_0_OUTPUT_TYPE
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_MAX
#undef FUSED_OP_0_OUTPUT_VAL_MAX
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_MIN
#undef FUSED_OP_0_OUTPUT_VAL_MIN
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_ONE
#undef FUSED_OP_0_OUTPUT_VAL_ONE
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_ZERO
#undef FUSED_OP_0_OUTPUT_VAL_ZERO
#endif
#ifdef TO_FUSED_OP_0_OUTPUT_TYPE
#undef TO_FUSED_OP_0_OUTPUT_TYPE
#endif
#ifdef TO_FUSED_OP_0_OUTPUT_TYPE_SAT
#undef TO_FUSED_OP_0_OUTPUT_TYPE_SAT
#endif
#ifdef AS_FUSED_OP_0_OUTPUT_TYPE
#undef AS_FUSED_OP_0_OUTPUT_TYPE
#endif
#ifdef FUSED_OP_0_OUTPUT_MAX_FUNC
#undef FUSED_OP_0_OUTPUT_MAX_FUNC
#endif
#ifdef FUSED_OP_0_OUTPUT_MIN_FUNC
#undef FUSED_OP_0_OUTPUT_MIN_FUNC
#endif
#ifdef FUSED_OP_0_OUTPUT_ABS_FUNC
#undef FUSED_OP_0_OUTPUT_ABS_FUNC
#endif
#ifdef FUSED_OP_0_OUTPUT_TYPE_SIZE
#undef FUSED_OP_0_OUTPUT_TYPE_SIZE
#endif
#ifdef FUSED_OP_0_OUTPUT_IS_FP
#undef FUSED_OP_0_OUTPUT_IS_FP
#endif
#ifdef FUSED_OP_0_OUTPUT_OFFSET
#undef FUSED_OP_0_OUTPUT_OFFSET
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZES_DATA
#undef FUSED_OP_0_OUTPUT_SIZES_DATA
#endif
#ifdef FUSED_OP_0_OUTPUT_PITCHES
#undef FUSED_OP_0_OUTPUT_PITCHES
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER
#undef FUSED_OP_0_OUTPUT_PAD_AFTER
#endif
#ifdef FUSED_OP0_DECLS
#undef FUSED_OP0_DECLS
#endif
#ifdef FUSED_OP0_ARGS
#undef FUSED_OP0_ARGS
#endif
#ifdef FUSED_OPS_DECLS
#undef FUSED_OPS_DECLS
#endif
#ifdef FUSED_OPS_ARGS
#undef FUSED_OPS_ARGS
#endif
#ifdef HAS_FUSED_OPS
#undef HAS_FUSED_OPS
#endif
#ifdef HAS_FUSED_OPS_DECLS
#undef HAS_FUSED_OPS_DECLS
#endif
#ifdef X_BLOCK_SIZE
#undef X_BLOCK_SIZE
#endif
#ifdef SLM_DIV_FACTOR
#undef SLM_DIV_FACTOR
#endif
#ifdef SUB_GROUP_SIZE
#undef SUB_GROUP_SIZE
#endif
#ifdef WORK_GROUP_SIZE
#undef WORK_GROUP_SIZE
#endif
#ifdef PADDED_INPUT
#undef PADDED_INPUT
#endif
#ifdef PADDED_OUTPUT
#undef PADDED_OUTPUT
#endif
#ifdef NON_UNIT_FUSED_OP_SPATIAL
#undef NON_UNIT_FUSED_OP_SPATIAL
#endif
#ifdef IC_BLOCKS
#undef IC_BLOCKS
#endif
#ifdef INPUT_LEFTOVERS
#undef INPUT_LEFTOVERS
#endif

//====================================================
// Kernel template: convolution_gpu_bfyx_f16_1x1 
// Kernel name: convolution_gpu_bfyx_f16_1x1_7824614608573370543_0_0
#define KERNEL(name) __kernel void convolution_gpu_bfyx_f16_1x1_7824614608573370543_0_0
#define KERNEL_ID convolution_gpu_bfyx_f16_1x1_7824614608573370543_0_0
#define FUNC(name)  _##name##_convolution_gpu_bfyx_f16_1x1_7824614608573370543_0_0
#define FUNC_CALL(name)  _##name##_convolution_gpu_bfyx_f16_1x1_7824614608573370543_0_0
#define CONST_ARRAY_DECL(name) __constant size_t  _##name##_convolution_gpu_bfyx_f16_1x1_7824614608573370543_0_0 []
#define CONST_ARRAY_REF(name)  _##name##_convolution_gpu_bfyx_f16_1x1_7824614608573370543_0_0
#define FP64_SUPPORTED 0
#define FP16_SUPPORTED 1
#define FP16_UNIT_USED 1
#define INT8_UNIT_USED 0
#define INT32_UNIT_USED 0
#define INT64_UNIT_USED 0
#define UINT8_UNIT_USED 0
#define UINT32_UNIT_USED 0
#define UNIT_TYPE half
#define UNIT_VAL_MAX HALF_MAX
#define UNIT_VAL_MIN -UNIT_VAL_MAX
#define UNIT_VAL_ONE 1.0h
#define UNIT_VAL_ZERO 0.0h
#define TO_UNIT_TYPE(v) convert_half(v)
#define TO_UNIT_TYPE_SAT(v) convert_half(v)
#define AS_UNIT_TYPE(v) as_half(v)
#define UNIT_MAX_FUNC fmax
#define UNIT_MIN_FUNC fmin
#define UNIT_ABS_FUNC fabs
#define UNIT_TYPE_SIZE 2
#define UNIT_IS_FP 1
#define NL_M as_float(0x0)/*0.000000e+00*/
#define NL_N as_float(0x0)/*0.000000e+00*/
#define ACTIVATION_FUNC_TYPE half
#define ACTIVATION_FUNC_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_VAL_MIN -ACTIVATION_FUNC_VAL_MAX
#define ACTIVATION_FUNC_VAL_ONE 1.0h
#define ACTIVATION_FUNC_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_MAX_FUNC fmax
#define ACTIVATION_FUNC_MIN_FUNC fmin
#define ACTIVATION_FUNC_ABS_FUNC fabs
#define ACTIVATION_FUNC_TYPE_SIZE 2
#define ACTIVATION_FUNC_IS_FP 1
#define ACTIVATION_PARAMS NL_M, NL_N
#define ACTIVATION_FUNC(input, m, n) input
#define ACTIVATION(input, params) ACTIVATION_FUNC(input, params)
#define INPUT0_SIZE_X 28
#define INPUT0_SIZE_Y 28
#define INPUT0_SIZE_Z 1
#define INPUT0_SIZE_W 1
#define INPUT0_SIZE_U 1
#define INPUT0_SIZE_V 1
#define INPUT0_FEATURE_NUM 128
#define INPUT0_BATCH_NUM 1
#define INPUT0_PAD_BEFORE_SIZE_X 0
#define INPUT0_PAD_BEFORE_SIZE_Y 0
#define INPUT0_PAD_BEFORE_SIZE_Z 0
#define INPUT0_PAD_BEFORE_SIZE_W 0
#define INPUT0_PAD_BEFORE_SIZE_U 0
#define INPUT0_PAD_BEFORE_SIZE_V 0
#define INPUT0_PAD_BEFORE_FEATURE_NUM 0
#define INPUT0_PAD_BEFORE_BATCH_NUM 0
#define INPUT0_PAD_AFTER_SIZE_X 0
#define INPUT0_PAD_AFTER_SIZE_Y 0
#define INPUT0_PAD_AFTER_SIZE_Z 0
#define INPUT0_PAD_AFTER_SIZE_W 0
#define INPUT0_PAD_AFTER_SIZE_U 0
#define INPUT0_PAD_AFTER_SIZE_V 0
#define INPUT0_PAD_AFTER_FEATURE_NUM 0
#define INPUT0_PAD_AFTER_BATCH_NUM 0
#define INPUT0_LENGTH 100352
#define INPUT0_X_PITCH 1
#define INPUT0_Y_PITCH 28
#define INPUT0_Z_PITCH 1
#define INPUT0_W_PITCH 1
#define INPUT0_U_PITCH 1
#define INPUT0_V_PITCH 1
#define INPUT0_FEATURE_PITCH 784
#define INPUT0_BATCH_PITCH 100352
#define INPUT0_GET_INDEX_SAFE(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX_SAFE(INPUT0, b, f, y, x)
#define INPUT0_GET_INDEX(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(INPUT0, b, f, y, x)
#define INPUT0_GET_INDEX_RAW(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(INPUT0, b, f, y, x)
#define INPUT0_VIEW_OFFSET 0
#define INPUT0_DIMS 4
#define INPUT0_SIMPLE 0
#define INPUT0_GROUPED 0
#define INPUT0_LAYOUT_B_FS_YX_FSV16 1
#define INPUT0_TYPE half
#define INPUT0_VAL_MAX HALF_MAX
#define INPUT0_VAL_MIN -INPUT0_VAL_MAX
#define INPUT0_VAL_ONE 1.0h
#define INPUT0_VAL_ZERO 0.0h
#define TO_INPUT0_TYPE(v) convert_half(v)
#define TO_INPUT0_TYPE_SAT(v) convert_half(v)
#define AS_INPUT0_TYPE(v) as_half(v)
#define INPUT0_MAX_FUNC fmax
#define INPUT0_MIN_FUNC fmin
#define INPUT0_ABS_FUNC fabs
#define INPUT0_TYPE_SIZE 2
#define INPUT0_IS_FP 1
#define INPUT0_OFFSET 0
#define INPUT0_SIZES_DATA { 28,28,128,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(INPUT0_SIZES) = INPUT0_SIZES_DATA;
#define INPUT0_SIZES CONST_ARRAY_REF(INPUT0_SIZES)
#define INPUT0_PITCHES (size_t []){ 1,28,784,100352,1,1,1,1,1, } 
#define INPUT0_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define INPUT0_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OUTPUT_SIZE_X 28
#define OUTPUT_SIZE_Y 28
#define OUTPUT_SIZE_Z 1
#define OUTPUT_SIZE_W 1
#define OUTPUT_SIZE_U 1
#define OUTPUT_SIZE_V 1
#define OUTPUT_FEATURE_NUM 512
#define OUTPUT_BATCH_NUM 1
#define OUTPUT_PAD_BEFORE_SIZE_X 0
#define OUTPUT_PAD_BEFORE_SIZE_Y 0
#define OUTPUT_PAD_BEFORE_SIZE_Z 0
#define OUTPUT_PAD_BEFORE_SIZE_W 0
#define OUTPUT_PAD_BEFORE_SIZE_U 0
#define OUTPUT_PAD_BEFORE_SIZE_V 0
#define OUTPUT_PAD_BEFORE_FEATURE_NUM 0
#define OUTPUT_PAD_BEFORE_BATCH_NUM 0
#define OUTPUT_PAD_AFTER_SIZE_X 0
#define OUTPUT_PAD_AFTER_SIZE_Y 0
#define OUTPUT_PAD_AFTER_SIZE_Z 0
#define OUTPUT_PAD_AFTER_SIZE_W 0
#define OUTPUT_PAD_AFTER_SIZE_U 0
#define OUTPUT_PAD_AFTER_SIZE_V 0
#define OUTPUT_PAD_AFTER_FEATURE_NUM 0
#define OUTPUT_PAD_AFTER_BATCH_NUM 0
#define OUTPUT_LENGTH 401408
#define OUTPUT_X_PITCH 1
#define OUTPUT_Y_PITCH 28
#define OUTPUT_Z_PITCH 1
#define OUTPUT_W_PITCH 1
#define OUTPUT_U_PITCH 1
#define OUTPUT_V_PITCH 1
#define OUTPUT_FEATURE_PITCH 784
#define OUTPUT_BATCH_PITCH 401408
#define OUTPUT_GET_INDEX_SAFE(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX_SAFE(OUTPUT, b, f, y, x)
#define OUTPUT_GET_INDEX(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(OUTPUT, b, f, y, x)
#define OUTPUT_GET_INDEX_RAW(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(OUTPUT, b, f, y, x)
#define OUTPUT_VIEW_OFFSET 0
#define OUTPUT_DIMS 4
#define OUTPUT_SIMPLE 0
#define OUTPUT_GROUPED 0
#define OUTPUT_LAYOUT_B_FS_YX_FSV16 1
#define OUTPUT_TYPE half
#define OUTPUT_VAL_MAX HALF_MAX
#define OUTPUT_VAL_MIN -OUTPUT_VAL_MAX
#define OUTPUT_VAL_ONE 1.0h
#define OUTPUT_VAL_ZERO 0.0h
#define TO_OUTPUT_TYPE(v) convert_half(v)
#define TO_OUTPUT_TYPE_SAT(v) convert_half(v)
#define AS_OUTPUT_TYPE(v) as_half(v)
#define OUTPUT_MAX_FUNC fmax
#define OUTPUT_MIN_FUNC fmin
#define OUTPUT_ABS_FUNC fabs
#define OUTPUT_TYPE_SIZE 2
#define OUTPUT_IS_FP 1
#define OUTPUT_OFFSET 0
#define OUTPUT_SIZES_DATA { 28,28,512,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(OUTPUT_SIZES) = OUTPUT_SIZES_DATA;
#define OUTPUT_SIZES CONST_ARRAY_REF(OUTPUT_SIZES)
#define OUTPUT_PITCHES (size_t []){ 1,28,784,401408,1,1,1,1,1, } 
#define OUTPUT_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OUTPUT_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OPTIONAL_SHAPE_INFO_ARG 
#define OPTIONAL_SHAPE_INFO_TENSOR 
#define FILTER_SIZE_X 1
#define FILTER_SIZE_Y 1
#define FILTER_SIZE_Z 1
#define FILTER_IFM_NUM 128
#define FILTER_OFM_NUM 512
#define FILTER_GROUPS_NUM 1
#define FILTER_X_PITCH 1
#define FILTER_Y_PITCH 1
#define FILTER_Z_PITCH 1
#define FILTER_IFM_PITCH 1
#define FILTER_OFM_PITCH 128
#define FILTER_GROUPS_PITCH 1
#define FILTER_LENGTH 65536
#define FILTER_VIEW_OFFSET 0
#define FILTER_DIMS 4
#define FILTER_SIMPLE 0
#define FILTER_GROUPED 0
#define FILTER_LAYOUT_OS_IS_YX_ISV16_OSV16 1
#define FILTER_TYPE half
#define FILTER_VAL_MAX HALF_MAX
#define FILTER_VAL_MIN -FILTER_VAL_MAX
#define FILTER_VAL_ONE 1.0h
#define FILTER_VAL_ZERO 0.0h
#define TO_FILTER_TYPE(v) convert_half(v)
#define TO_FILTER_TYPE_SAT(v) convert_half(v)
#define AS_FILTER_TYPE(v) as_half(v)
#define FILTER_MAX_FUNC fmax
#define FILTER_MIN_FUNC fmin
#define FILTER_ABS_FUNC fabs
#define FILTER_TYPE_SIZE 2
#define FILTER_IS_FP 1
#define FILTER_OFFSET 0
#define FILTER_SIZES_DATA { 1,1,128,512,1,1,1,1,1, } 
CONST_ARRAY_DECL(FILTER_SIZES) = FILTER_SIZES_DATA;
#define FILTER_SIZES CONST_ARRAY_REF(FILTER_SIZES)
#define FILTER_PITCHES (size_t []){ 1,1,1,128,1,1,1,1,1, } 
#define FILTER_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FILTER_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define BIAS_TERM 1
#define BIAS_SIZE_X 1
#define BIAS_SIZE_Y 1
#define BIAS_SIZE_Z 1
#define BIAS_SIZE_W 1
#define BIAS_SIZE_U 1
#define BIAS_SIZE_V 1
#define BIAS_FEATURE_NUM 512
#define BIAS_BATCH_NUM 1
#define BIAS_PAD_BEFORE_SIZE_X 0
#define BIAS_PAD_BEFORE_SIZE_Y 0
#define BIAS_PAD_BEFORE_SIZE_Z 0
#define BIAS_PAD_BEFORE_SIZE_W 0
#define BIAS_PAD_BEFORE_SIZE_U 0
#define BIAS_PAD_BEFORE_SIZE_V 0
#define BIAS_PAD_BEFORE_FEATURE_NUM 0
#define BIAS_PAD_BEFORE_BATCH_NUM 0
#define BIAS_PAD_AFTER_SIZE_X 0
#define BIAS_PAD_AFTER_SIZE_Y 0
#define BIAS_PAD_AFTER_SIZE_Z 0
#define BIAS_PAD_AFTER_SIZE_W 0
#define BIAS_PAD_AFTER_SIZE_U 0
#define BIAS_PAD_AFTER_SIZE_V 0
#define BIAS_PAD_AFTER_FEATURE_NUM 0
#define BIAS_PAD_AFTER_BATCH_NUM 0
#define BIAS_LENGTH 512
#define BIAS_X_PITCH 1
#define BIAS_Y_PITCH 1
#define BIAS_Z_PITCH 1
#define BIAS_W_PITCH 1
#define BIAS_U_PITCH 1
#define BIAS_V_PITCH 1
#define BIAS_FEATURE_PITCH 1
#define BIAS_BATCH_PITCH 512
#define BIAS_GET_INDEX_SAFE(b, f, y, x) ((0 + (f)) % 512)
#define BIAS_GET_INDEX(b, f, y, x) (0 + (f))
#define BIAS_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(BIAS, b, f, y, x)
#define BIAS_VIEW_OFFSET 0
#define BIAS_DIMS 2
#define BIAS_SIMPLE 1
#define BIAS_GROUPED 0
#define BIAS_LAYOUT_BF 1
#define BIAS_TYPE half
#define BIAS_VAL_MAX HALF_MAX
#define BIAS_VAL_MIN -BIAS_VAL_MAX
#define BIAS_VAL_ONE 1.0h
#define BIAS_VAL_ZERO 0.0h
#define TO_BIAS_TYPE(v) convert_half(v)
#define TO_BIAS_TYPE_SAT(v) convert_half(v)
#define AS_BIAS_TYPE(v) as_half(v)
#define BIAS_MAX_FUNC fmax
#define BIAS_MIN_FUNC fmin
#define BIAS_ABS_FUNC fabs
#define BIAS_TYPE_SIZE 2
#define BIAS_IS_FP 1
#define BIAS_OFFSET 0
#define BIAS_SIZES_DATA { 512,1,1,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(BIAS_SIZES) = BIAS_SIZES_DATA;
#define BIAS_SIZES CONST_ARRAY_REF(BIAS_SIZES)
#define BIAS_PITCHES (size_t []){ 1,512,1,1,1,1,1,1,1, } 
#define BIAS_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define BIAS_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define BIAS_PER_OUTPUT 0
#define BIAS_PER_OFM 1
#define STRIDE_SIZE_X 1
#define STRIDE_SIZE_Y 1
#define STRIDE_SIZE_Z 1
#define PADDING_SIZE_X 0
#define PADDING_SIZE_Y 0
#define PADDING_SIZE_Z 0
#define DILATION_SIZE_X 1
#define DILATION_SIZE_Y 1
#define DILATION_SIZE_Z 1
#define FILTER_ARRAY_NUM 1
#define INPUT0_OFFSET_WITH_PADDING 0
#define GROUPED 0
#define SKIP_BATCH 1
#define FUSED_OP0_LOAD_VEC \
	MAKE_VECTOR_TYPE(FUSED_OP_0_INPUT0_TYPE,8) eltwise0_data0 = as_half8( _sub_group_block_read_us8((const __global ushort*)(eltwise0_input0 + FUSED_OP_0_INPUT0_GET_INDEX_SAFE(0,(feature_block * 16),y,x))));
#define FUSED_OP0_ACTION_VEC \
	half8 dst_out_0_tmp = dst+eltwise0_data0;\
	half8 dst_out_0 = convert_half8(dst_out_0_tmp);
#define FUSED_OP1_LOAD_VEC 
#define ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE half
#define ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_MIN -ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_MAX
#define ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_ONE 1.0h
#define ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_FUSED_OP1_VEC_MAX_FUNC fmax
#define ACTIVATION_FUNC_FUSED_OP1_VEC_MIN_FUNC fmin
#define ACTIVATION_FUNC_FUSED_OP1_VEC_ABS_FUNC fabs
#define ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE_SIZE 2
#define ACTIVATION_FUNC_FUSED_OP1_VEC_IS_FP 1
#define ACTIVATION_PARAMS_FUSED_OP1_VEC NL_M_FUSED_OP1_VEC, NL_N_FUSED_OP1_VEC
#define ACTIVATION_FUNC_FUSED_OP1_VEC(input, m, n) (ACTIVATION_FUNC_FUSED_OP1_VEC_MAX_FUNC(ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_ZERO, input))
#define ACTIVATION_FUSED_OP1_VEC(input, params) ACTIVATION_FUNC_FUSED_OP1_VEC(input, params)
#define FUSED_OP1_ACTION_VEC \
	half8 dst_out_1 = convert_half8(dst_out_0);\
	dst_out_1 = ACTIVATION_FUNC_FUSED_OP1_VEC(dst_out_1, (half8)(as_float(0x0)/*0.000000e+00*/),(half8)(as_float(0x0)/*0.000000e+00*/));
#define FUSED_OPS_VEC \
	FUSED_OP0_LOAD_VEC\
	FUSED_OP0_ACTION_VEC\
	FUSED_OP1_LOAD_VEC\
	FUSED_OP1_ACTION_VEC
#define FUSED_OPS_PRELOAD_VEC \
	FUSED_OP0_LOAD_VEC\
	FUSED_OP1_LOAD_VEC
#define FUSED_OPS_CALC_VEC \
	FUSED_OP0_ACTION_VEC\
	FUSED_OP1_ACTION_VEC
#define FUSED_OPS_RESULT_VEC dst_out_1
#define FUSED_OPS_CAN_USE_PRELOAD_VEC 1
#define FUSED_OP0_LOAD_SCALAR \
	FUSED_OP_0_INPUT0_TYPE eltwise0_data0 = (half)(as_half( _sub_group_block_read_us((const __global ushort*)(eltwise0_input0 + FUSED_OP_0_INPUT0_GET_INDEX_SAFE(0,(feature_block * 16),yi,xi)))));
#define FUSED_OP0_ACTION_SCALAR \
	half dst_i__out_0_tmp = dst[i]+eltwise0_data0;\
	half dst_i__out_0 = convert_half(dst_i__out_0_tmp);
#define FUSED_OP1_LOAD_SCALAR 
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE half
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_MIN -ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_MAX
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_ONE 1.0h
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_MAX_FUNC fmax
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_MIN_FUNC fmin
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_ABS_FUNC fabs
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE_SIZE 2
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_IS_FP 1
#define ACTIVATION_PARAMS_FUSED_OP1_SCALAR NL_M_FUSED_OP1_SCALAR, NL_N_FUSED_OP1_SCALAR
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR(input, m, n) (ACTIVATION_FUNC_FUSED_OP1_SCALAR_MAX_FUNC(ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_ZERO, input))
#define ACTIVATION_FUSED_OP1_SCALAR(input, params) ACTIVATION_FUNC_FUSED_OP1_SCALAR(input, params)
#define FUSED_OP1_ACTION_SCALAR \
	half dst_i__out_1 = convert_half(dst_i__out_0);\
	dst_i__out_1 = ACTIVATION_FUNC_FUSED_OP1_SCALAR(dst_i__out_1, (half)(as_float(0x0)/*0.000000e+00*/),(half)(as_float(0x0)/*0.000000e+00*/));
#define FUSED_OPS_SCALAR \
	FUSED_OP0_LOAD_SCALAR\
	FUSED_OP0_ACTION_SCALAR\
	FUSED_OP1_LOAD_SCALAR\
	FUSED_OP1_ACTION_SCALAR
#define FUSED_OPS_PRELOAD_SCALAR \
	FUSED_OP0_LOAD_SCALAR\
	FUSED_OP1_LOAD_SCALAR
#define FUSED_OPS_CALC_SCALAR \
	FUSED_OP0_ACTION_SCALAR\
	FUSED_OP1_ACTION_SCALAR
#define FUSED_OPS_RESULT_SCALAR dst_i__out_1
#define FUSED_OPS_CAN_USE_PRELOAD_SCALAR 1
#define FUSED_OP0_LOAD_SCALAR_B1 \
	FUSED_OP_0_INPUT0_TYPE eltwise0_data0 = (half)(as_half( _sub_group_block_read_us((const __global ushort*)(eltwise0_input0 + FUSED_OP_0_INPUT0_GET_INDEX_SAFE(0,(feature_block * 16),0,0)))));
#define FUSED_OP0_ACTION_SCALAR_B1 \
	half dst_out_0_tmp = dst+eltwise0_data0;\
	half dst_out_0 = convert_half(dst_out_0_tmp);
#define FUSED_OP1_LOAD_SCALAR_B1 
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE half
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_MIN -ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_MAX
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_ONE 1.0h
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_MAX_FUNC fmax
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_MIN_FUNC fmin
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_ABS_FUNC fabs
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE_SIZE 2
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_IS_FP 1
#define ACTIVATION_PARAMS_FUSED_OP1_SCALAR_B1 NL_M_FUSED_OP1_SCALAR_B1, NL_N_FUSED_OP1_SCALAR_B1
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1(input, m, n) (ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_MAX_FUNC(ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_ZERO, input))
#define ACTIVATION_FUSED_OP1_SCALAR_B1(input, params) ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1(input, params)
#define FUSED_OP1_ACTION_SCALAR_B1 \
	half dst_out_1 = convert_half(dst_out_0);\
	dst_out_1 = ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1(dst_out_1, (half)(as_float(0x0)/*0.000000e+00*/),(half)(as_float(0x0)/*0.000000e+00*/));
#define FUSED_OPS_SCALAR_B1 \
	FUSED_OP0_LOAD_SCALAR_B1\
	FUSED_OP0_ACTION_SCALAR_B1\
	FUSED_OP1_LOAD_SCALAR_B1\
	FUSED_OP1_ACTION_SCALAR_B1
#define FUSED_OPS_PRELOAD_SCALAR_B1 \
	FUSED_OP0_LOAD_SCALAR_B1\
	FUSED_OP1_LOAD_SCALAR_B1
#define FUSED_OPS_CALC_SCALAR_B1 \
	FUSED_OP0_ACTION_SCALAR_B1\
	FUSED_OP1_ACTION_SCALAR_B1
#define FUSED_OPS_RESULT_SCALAR_B1 dst_out_1
#define FUSED_OPS_CAN_USE_PRELOAD_SCALAR_B1 1
#define FUSED_OP_0_INPUT0_SIZE_X 28
#define FUSED_OP_0_INPUT0_SIZE_Y 28
#define FUSED_OP_0_INPUT0_SIZE_Z 1
#define FUSED_OP_0_INPUT0_SIZE_W 1
#define FUSED_OP_0_INPUT0_SIZE_U 1
#define FUSED_OP_0_INPUT0_SIZE_V 1
#define FUSED_OP_0_INPUT0_FEATURE_NUM 512
#define FUSED_OP_0_INPUT0_BATCH_NUM 1
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_X 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Y 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Z 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_W 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_U 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_V 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_FEATURE_NUM 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_BATCH_NUM 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_X 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Y 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Z 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_W 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_U 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_V 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_FEATURE_NUM 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_BATCH_NUM 0
#define FUSED_OP_0_INPUT0_LENGTH 401408
#define FUSED_OP_0_INPUT0_X_PITCH 1
#define FUSED_OP_0_INPUT0_Y_PITCH 28
#define FUSED_OP_0_INPUT0_Z_PITCH 1
#define FUSED_OP_0_INPUT0_W_PITCH 1
#define FUSED_OP_0_INPUT0_U_PITCH 1
#define FUSED_OP_0_INPUT0_V_PITCH 1
#define FUSED_OP_0_INPUT0_FEATURE_PITCH 784
#define FUSED_OP_0_INPUT0_BATCH_PITCH 401408
#define FUSED_OP_0_INPUT0_GET_INDEX_SAFE(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX_SAFE(FUSED_OP_0_INPUT0, b, f, y, x)
#define FUSED_OP_0_INPUT0_GET_INDEX(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(FUSED_OP_0_INPUT0, b, f, y, x)
#define FUSED_OP_0_INPUT0_GET_INDEX_RAW(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(FUSED_OP_0_INPUT0, b, f, y, x)
#define FUSED_OP_0_INPUT0_VIEW_OFFSET 0
#define FUSED_OP_0_INPUT0_DIMS 4
#define FUSED_OP_0_INPUT0_SIMPLE 0
#define FUSED_OP_0_INPUT0_GROUPED 0
#define FUSED_OP_0_INPUT0_LAYOUT_B_FS_YX_FSV16 1
#define FUSED_OP_0_INPUT0_TYPE half
#define FUSED_OP_0_INPUT0_VAL_MAX HALF_MAX
#define FUSED_OP_0_INPUT0_VAL_MIN -FUSED_OP_0_INPUT0_VAL_MAX
#define FUSED_OP_0_INPUT0_VAL_ONE 1.0h
#define FUSED_OP_0_INPUT0_VAL_ZERO 0.0h
#define TO_FUSED_OP_0_INPUT0_TYPE(v) convert_half(v)
#define TO_FUSED_OP_0_INPUT0_TYPE_SAT(v) convert_half(v)
#define AS_FUSED_OP_0_INPUT0_TYPE(v) as_half(v)
#define FUSED_OP_0_INPUT0_MAX_FUNC fmax
#define FUSED_OP_0_INPUT0_MIN_FUNC fmin
#define FUSED_OP_0_INPUT0_ABS_FUNC fabs
#define FUSED_OP_0_INPUT0_TYPE_SIZE 2
#define FUSED_OP_0_INPUT0_IS_FP 1
#define FUSED_OP_0_INPUT0_OFFSET 0
#define FUSED_OP_0_INPUT0_SIZES_DATA { 28,28,512,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(FUSED_OP_0_INPUT0_SIZES) = FUSED_OP_0_INPUT0_SIZES_DATA;
#define FUSED_OP_0_INPUT0_SIZES CONST_ARRAY_REF(FUSED_OP_0_INPUT0_SIZES)
#define FUSED_OP_0_INPUT0_PITCHES (size_t []){ 1,28,784,401408,1,1,1,1,1, } 
#define FUSED_OP_0_INPUT0_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP_0_INPUT0_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP_0_OUTPUT_SIZE_X 28
#define FUSED_OP_0_OUTPUT_SIZE_Y 28
#define FUSED_OP_0_OUTPUT_SIZE_Z 1
#define FUSED_OP_0_OUTPUT_SIZE_W 1
#define FUSED_OP_0_OUTPUT_SIZE_U 1
#define FUSED_OP_0_OUTPUT_SIZE_V 1
#define FUSED_OP_0_OUTPUT_FEATURE_NUM 512
#define FUSED_OP_0_OUTPUT_BATCH_NUM 1
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_X 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Y 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Z 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_W 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_U 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_V 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_FEATURE_NUM 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_BATCH_NUM 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_X 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Y 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Z 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_W 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_U 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_V 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_FEATURE_NUM 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_BATCH_NUM 0
#define FUSED_OP_0_OUTPUT_LENGTH 401408
#define FUSED_OP_0_OUTPUT_X_PITCH 1
#define FUSED_OP_0_OUTPUT_Y_PITCH 28
#define FUSED_OP_0_OUTPUT_Z_PITCH 1
#define FUSED_OP_0_OUTPUT_W_PITCH 1
#define FUSED_OP_0_OUTPUT_U_PITCH 1
#define FUSED_OP_0_OUTPUT_V_PITCH 1
#define FUSED_OP_0_OUTPUT_FEATURE_PITCH 784
#define FUSED_OP_0_OUTPUT_BATCH_PITCH 401408
#define FUSED_OP_0_OUTPUT_GET_INDEX_SAFE(b, f, y, x) GET_DATA_INDEX_SAFE(FUSED_OP_0_OUTPUT, b, f, y, x)
#define FUSED_OP_0_OUTPUT_GET_INDEX(b, f, y, x) GET_DATA_INDEX(FUSED_OP_0_OUTPUT, b, f, y, x)
#define FUSED_OP_0_OUTPUT_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(FUSED_OP_0_OUTPUT, b, f, y, x)
#define FUSED_OP_0_OUTPUT_VIEW_OFFSET 0
#define FUSED_OP_0_OUTPUT_DIMS 4
#define FUSED_OP_0_OUTPUT_SIMPLE 1
#define FUSED_OP_0_OUTPUT_GROUPED 0
#define FUSED_OP_0_OUTPUT_LAYOUT_BFYX 1
#define FUSED_OP_0_OUTPUT_TYPE half
#define FUSED_OP_0_OUTPUT_VAL_MAX HALF_MAX
#define FUSED_OP_0_OUTPUT_VAL_MIN -FUSED_OP_0_OUTPUT_VAL_MAX
#define FUSED_OP_0_OUTPUT_VAL_ONE 1.0h
#define FUSED_OP_0_OUTPUT_VAL_ZERO 0.0h
#define TO_FUSED_OP_0_OUTPUT_TYPE(v) convert_half(v)
#define TO_FUSED_OP_0_OUTPUT_TYPE_SAT(v) convert_half(v)
#define AS_FUSED_OP_0_OUTPUT_TYPE(v) as_half(v)
#define FUSED_OP_0_OUTPUT_MAX_FUNC fmax
#define FUSED_OP_0_OUTPUT_MIN_FUNC fmin
#define FUSED_OP_0_OUTPUT_ABS_FUNC fabs
#define FUSED_OP_0_OUTPUT_TYPE_SIZE 2
#define FUSED_OP_0_OUTPUT_IS_FP 1
#define FUSED_OP_0_OUTPUT_OFFSET 0
#define FUSED_OP_0_OUTPUT_SIZES_DATA { 28,28,512,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(FUSED_OP_0_OUTPUT_SIZES) = FUSED_OP_0_OUTPUT_SIZES_DATA;
#define FUSED_OP_0_OUTPUT_SIZES CONST_ARRAY_REF(FUSED_OP_0_OUTPUT_SIZES)
#define FUSED_OP_0_OUTPUT_PITCHES (size_t []){ 1,28,784,401408,1,1,1,1,1, } 
#define FUSED_OP_0_OUTPUT_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP_0_OUTPUT_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP0_DECLS \
	const __global half* eltwise0_input0
#define FUSED_OP0_ARGS \
	eltwise0_input0
#define FUSED_OP_1_OUTPUT_SIZE_X 28
#define FUSED_OP_1_OUTPUT_SIZE_Y 28
#define FUSED_OP_1_OUTPUT_SIZE_Z 1
#define FUSED_OP_1_OUTPUT_SIZE_W 1
#define FUSED_OP_1_OUTPUT_SIZE_U 1
#define FUSED_OP_1_OUTPUT_SIZE_V 1
#define FUSED_OP_1_OUTPUT_FEATURE_NUM 512
#define FUSED_OP_1_OUTPUT_BATCH_NUM 1
#define FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_X 0
#define FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_Y 0
#define FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_Z 0
#define FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_W 0
#define FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_U 0
#define FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_V 0
#define FUSED_OP_1_OUTPUT_PAD_BEFORE_FEATURE_NUM 0
#define FUSED_OP_1_OUTPUT_PAD_BEFORE_BATCH_NUM 0
#define FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_X 0
#define FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_Y 0
#define FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_Z 0
#define FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_W 0
#define FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_U 0
#define FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_V 0
#define FUSED_OP_1_OUTPUT_PAD_AFTER_FEATURE_NUM 0
#define FUSED_OP_1_OUTPUT_PAD_AFTER_BATCH_NUM 0
#define FUSED_OP_1_OUTPUT_LENGTH 401408
#define FUSED_OP_1_OUTPUT_X_PITCH 1
#define FUSED_OP_1_OUTPUT_Y_PITCH 28
#define FUSED_OP_1_OUTPUT_Z_PITCH 1
#define FUSED_OP_1_OUTPUT_W_PITCH 1
#define FUSED_OP_1_OUTPUT_U_PITCH 1
#define FUSED_OP_1_OUTPUT_V_PITCH 1
#define FUSED_OP_1_OUTPUT_FEATURE_PITCH 784
#define FUSED_OP_1_OUTPUT_BATCH_PITCH 401408
#define FUSED_OP_1_OUTPUT_GET_INDEX_SAFE(b, f, y, x) GET_DATA_INDEX_SAFE(FUSED_OP_1_OUTPUT, b, f, y, x)
#define FUSED_OP_1_OUTPUT_GET_INDEX(b, f, y, x) GET_DATA_INDEX(FUSED_OP_1_OUTPUT, b, f, y, x)
#define FUSED_OP_1_OUTPUT_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(FUSED_OP_1_OUTPUT, b, f, y, x)
#define FUSED_OP_1_OUTPUT_VIEW_OFFSET 0
#define FUSED_OP_1_OUTPUT_DIMS 4
#define FUSED_OP_1_OUTPUT_SIMPLE 1
#define FUSED_OP_1_OUTPUT_GROUPED 0
#define FUSED_OP_1_OUTPUT_LAYOUT_BFYX 1
#define FUSED_OP_1_OUTPUT_TYPE half
#define FUSED_OP_1_OUTPUT_VAL_MAX HALF_MAX
#define FUSED_OP_1_OUTPUT_VAL_MIN -FUSED_OP_1_OUTPUT_VAL_MAX
#define FUSED_OP_1_OUTPUT_VAL_ONE 1.0h
#define FUSED_OP_1_OUTPUT_VAL_ZERO 0.0h
#define TO_FUSED_OP_1_OUTPUT_TYPE(v) convert_half(v)
#define TO_FUSED_OP_1_OUTPUT_TYPE_SAT(v) convert_half(v)
#define AS_FUSED_OP_1_OUTPUT_TYPE(v) as_half(v)
#define FUSED_OP_1_OUTPUT_MAX_FUNC fmax
#define FUSED_OP_1_OUTPUT_MIN_FUNC fmin
#define FUSED_OP_1_OUTPUT_ABS_FUNC fabs
#define FUSED_OP_1_OUTPUT_TYPE_SIZE 2
#define FUSED_OP_1_OUTPUT_IS_FP 1
#define FUSED_OP_1_OUTPUT_OFFSET 0
#define FUSED_OP_1_OUTPUT_SIZES_DATA { 28,28,512,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(FUSED_OP_1_OUTPUT_SIZES) = FUSED_OP_1_OUTPUT_SIZES_DATA;
#define FUSED_OP_1_OUTPUT_SIZES CONST_ARRAY_REF(FUSED_OP_1_OUTPUT_SIZES)
#define FUSED_OP_1_OUTPUT_PITCHES (size_t []){ 1,28,784,401408,1,1,1,1,1, } 
#define FUSED_OP_1_OUTPUT_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP_1_OUTPUT_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP1_DECLS 
#define FUSED_OP1_ARGS 
#define FUSED_OPS_DECLS \
	FUSED_OP0_DECLS
#define FUSED_OPS_ARGS \
	FUSED_OP0_ARGS
#define HAS_FUSED_OPS 1
#define HAS_FUSED_OPS_DECLS 1
#define X_BLOCK_SIZE 8
#define SLM_DIV_FACTOR 1
#define SUB_GROUP_SIZE 16
#define WORK_GROUP_SIZE 16
#define PADDED_INPUT 0
#define PADDED_OUTPUT 0
#define NON_UNIT_FUSED_OP_SPATIAL 1
#define IC_BLOCKS 8
#define INPUT_LEFTOVERS 0


#ifndef UNIT_TYPE
#error Unit type not defined.
#endif
#if UNIT_TYPE_SIZE == 2
#define UNIT_BLOCK_RW_TYPE ushort
#define UNIT_BLOCK_READ_FUNC _sub_group_block_read_us
#define UNIT_BLOCK_WRITE_FUNC _sub_group_block_write_us
#elif UNIT_TYPE_SIZE == 4
#define UNIT_BLOCK_RW_TYPE uint
#define UNIT_BLOCK_READ_FUNC _sub_group_block_read
#define UNIT_BLOCK_WRITE_FUNC _sub_group_block_write
#else
#error Unsupported unit type for block read/write.
#endif
#define UNIT_TYPE2 MAKE_VECTOR_TYPE(UNIT_TYPE, 2)
#define UNIT_TYPE4 MAKE_VECTOR_TYPE(UNIT_TYPE, 4)
#define UNIT_TYPE8 MAKE_VECTOR_TYPE(UNIT_TYPE, 8)
#define UNIT_BLOCK_RW_TYPE2 MAKE_VECTOR_TYPE(UNIT_BLOCK_RW_TYPE, 2)
#define UNIT_BLOCK_RW_TYPE4 MAKE_VECTOR_TYPE(UNIT_BLOCK_RW_TYPE, 4)
#define UNIT_BLOCK_RW_TYPE8 MAKE_VECTOR_TYPE(UNIT_BLOCK_RW_TYPE, 8)
#define UNIT_BLOCK_READ_FUNC2 CAT(UNIT_BLOCK_READ_FUNC, 2)
#define UNIT_BLOCK_READ_FUNC4 CAT(UNIT_BLOCK_READ_FUNC, 4)
#define UNIT_BLOCK_READ_FUNC8 CAT(UNIT_BLOCK_READ_FUNC, 8)
#define UNIT_BLOCK_WRITE_FUNC2 CAT(UNIT_BLOCK_WRITE_FUNC, 2)
#define UNIT_BLOCK_WRITE_FUNC4 CAT(UNIT_BLOCK_WRITE_FUNC, 4)
#define UNIT_BLOCK_WRITE_FUNC8 CAT(UNIT_BLOCK_WRITE_FUNC, 8)
#define UNIT_BLOCK_READ(ptr, offset) AS_TYPE(UNIT_TYPE, UNIT_BLOCK_READ_FUNC( (const __global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset)))
#define UNIT_BLOCK_READ2(ptr, offset) AS_TYPE(UNIT_TYPE2, UNIT_BLOCK_READ_FUNC2((const __global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset)))
#define UNIT_BLOCK_READ4(ptr, offset) AS_TYPE(UNIT_TYPE4, UNIT_BLOCK_READ_FUNC4((const __global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset)))
#define UNIT_BLOCK_READ8(ptr, offset) AS_TYPE(UNIT_TYPE8, UNIT_BLOCK_READ_FUNC8((const __global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset)))
#define UNIT_BLOCK_WRITE(ptr, offset, val) UNIT_BLOCK_WRITE_FUNC( (__global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset), AS_TYPE(UNIT_BLOCK_RW_TYPE, val))
#define UNIT_BLOCK_WRITE2(ptr, offset, val) UNIT_BLOCK_WRITE_FUNC2((__global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset), AS_TYPE(UNIT_BLOCK_RW_TYPE2, val))
#define UNIT_BLOCK_WRITE4(ptr, offset, val) UNIT_BLOCK_WRITE_FUNC4((__global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset), AS_TYPE(UNIT_BLOCK_RW_TYPE4, val))
#define UNIT_BLOCK_WRITE8(ptr, offset, val) UNIT_BLOCK_WRITE_FUNC8((__global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset), AS_TYPE(UNIT_BLOCK_RW_TYPE8, val))
#if X_BLOCK_SIZE > 1
# define GET_SRC(data, id) AS_TYPE(MAKE_VECTOR_TYPE(UNIT_TYPE, X_BLOCK_SIZE), \
 _sub_group_shuffle( \
 AS_TYPE(MAKE_VECTOR_TYPE(UNIT_BLOCK_RW_TYPE, X_BLOCK_SIZE), data), \
 id))
#else
# define GET_SRC(data, id) AS_TYPE(UNIT_TYPE, _sub_group_shuffle(AS_TYPE(UNIT_BLOCK_RW_TYPE, data), id))
#endif
#define FEATURE_SLICE_SIZE 16
#if X_BLOCK_SIZE > 1
# define UNIT_BLOCK_READ_VEC(ptr, offset) CAT(UNIT_BLOCK_READ, X_BLOCK_SIZE)(ptr, offset)
# define UNIT_BLOCK_WRITE_VEC(ptr, offset, val) CAT(UNIT_BLOCK_WRITE, X_BLOCK_SIZE)(ptr, offset, val)
#endif
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, SUB_GROUP_SIZE * SLM_DIV_FACTOR, 1)))
KERNEL(convolution_b_fs_yx_fsv16_1x1)(
 OPTIONAL_SHAPE_INFO_ARG
 __global INPUT0_TYPE* input,
 __global OUTPUT_TYPE* output,
 __global FILTER_TYPE* weights
#if BIAS_TERM
 , __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
 , FUSED_OPS_DECLS
#endif
) {
#if X_BLOCK_SIZE > 1
 const uint xy = (int)get_global_id(0);
 const uint x = (xy * X_BLOCK_SIZE) % OUTPUT_SIZE_X;
 const uint y = (xy * X_BLOCK_SIZE) / OUTPUT_SIZE_X;
 const uint input_x = x;
 const uint input_y = y;
#endif
 const uint b = (int)get_global_id(2);
 const uint sglid = (int)get_sub_group_local_id();
 const uint lid1 = (int)get_local_id(1);
 const uint feature_per_wg = (int)get_local_size(1) / SLM_DIV_FACTOR;
 const uint feature_sub_block = lid1 / feature_per_wg;
 const uint feature_block = (int)get_group_id(1);
 const uint input_x_pitch = FEATURE_SLICE_SIZE;
 const uint input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
 const uint input_fs_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
 const uint input_total_f_size = INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
 const uint input_b_pitch = input_fs_pitch * ((input_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
 const uint input_fs_pad_before = INPUT0_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;
 const uint input_offset = b * input_b_pitch +
 input_fs_pad_before * input_fs_pitch +
 INPUT0_PAD_BEFORE_SIZE_Y * input_y_pitch +
 INPUT0_PAD_BEFORE_SIZE_X * input_x_pitch;
 const uint output_x_pitch = FEATURE_SLICE_SIZE;
 const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
 const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
 const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
 const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
 const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;
 const uint output_offset = b * output_b_pitch +
 (feature_block + output_fs_pad_before) * output_fs_pitch +
 (OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
 (OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;
 const uint filter_isv_pitch = FEATURE_SLICE_SIZE;
 const uint filter_x_pitch = FEATURE_SLICE_SIZE * FEATURE_SLICE_SIZE;
 const uint filter_y_pitch = filter_x_pitch * FILTER_SIZE_X;
 const uint filter_is_pitch = filter_y_pitch * FILTER_SIZE_Y;
 const uint filter_os_pitch = filter_is_pitch * ((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
 const uint filter_offset = feature_block * filter_os_pitch;
#if X_BLOCK_SIZE > 1
 typedef MAKE_VECTOR_TYPE(UNIT_TYPE, X_BLOCK_SIZE) vec_t;
#else
 typedef UNIT_TYPE vec_t;
#endif
#if BIAS_TERM
#if SLM_DIV_FACTOR == 1
 vec_t dst = (vec_t)(UNIT_BLOCK_READ(biases, feature_block * FEATURE_SLICE_SIZE));
#else
 vec_t dst;
 if (feature_sub_block == 0) {
 dst = (vec_t)(UNIT_BLOCK_READ(biases, feature_block * FEATURE_SLICE_SIZE));
 } else {
 dst = UNIT_VAL_ZERO;
 }
#endif
#else
 vec_t dst = UNIT_VAL_ZERO;
#endif
#if SLM_DIV_FACTOR > 1
 __local vec_t partial_summ[WORK_GROUP_SIZE];
 for (uint k = feature_sub_block * IC_BLOCKS / SLM_DIV_FACTOR; k < (feature_sub_block + 1) * IC_BLOCKS / SLM_DIV_FACTOR; k++)
 {
#else
 for (uint k = 0; k < IC_BLOCKS; k++)
 {
#endif
 vec_t src = 0;
 if (INPUT_LEFTOVERS && ((k + 1) * FEATURE_SLICE_SIZE >= INPUT0_FEATURE_NUM))
 {
 if (k * FEATURE_SLICE_SIZE + sglid < INPUT0_FEATURE_NUM)
 {
#if X_BLOCK_SIZE > 1
 __attribute__((opencl_unroll_hint(X_BLOCK_SIZE)))
 for (int i = 0; i < X_BLOCK_SIZE; i++)
 {
 const uint xb = (x + i) % INPUT0_SIZE_X;
 const uint yb = y + (x + i) / INPUT0_SIZE_X;
 const uint input_idx = input_offset + k * input_fs_pitch + yb * input_y_pitch + xb * input_x_pitch;
 src[i] = input[input_idx + sglid];
 }
#else
 src = input[input_offset + k * input_fs_pitch + sglid];
#endif
 }
 }
 else
 {
#if PADDED_INPUT
#if X_BLOCK_SIZE > 1
 __attribute__((opencl_unroll_hint(X_BLOCK_SIZE)))
 for (int i = 0; i < X_BLOCK_SIZE; i++)
 {
 const uint xb = (x + i) % INPUT0_SIZE_X;
 const uint yb = y + (x + i) / INPUT0_SIZE_X;
 const uint input_idx = input_offset + k * input_fs_pitch + yb * input_y_pitch + xb * input_x_pitch;
 src[i] = UNIT_BLOCK_READ(input, input_idx);
 }
#else
 src = UNIT_BLOCK_READ(input, input_offset + k * input_fs_pitch);
#endif
#else
#if X_BLOCK_SIZE > 1
 src = UNIT_BLOCK_READ_VEC(input, input_offset + k * input_fs_pitch + input_y * input_y_pitch + input_x * input_x_pitch);
#else
 src = UNIT_BLOCK_READ(input, input_offset + k * input_fs_pitch);
#endif
#endif
 }
 UNIT_TYPE8 wei0 = UNIT_BLOCK_READ8(weights, filter_offset + k * filter_is_pitch);
 UNIT_TYPE8 wei1 = UNIT_BLOCK_READ8(weights, filter_offset + k * filter_is_pitch + 8 * filter_isv_pitch);
 const vec_t src0 = GET_SRC(src, 0);
 const vec_t src1 = GET_SRC(src, 1);
 const vec_t src2 = GET_SRC(src, 2);
 const vec_t src3 = GET_SRC(src, 3);
 const vec_t src4 = GET_SRC(src, 4);
 const vec_t src5 = GET_SRC(src, 5);
 const vec_t src6 = GET_SRC(src, 6);
 const vec_t src7 = GET_SRC(src, 7);
 const vec_t src8 = GET_SRC(src, 8);
 const vec_t src9 = GET_SRC(src, 9);
 const vec_t src10 = GET_SRC(src, 10);
 const vec_t src11 = GET_SRC(src, 11);
 const vec_t src12 = GET_SRC(src, 12);
 const vec_t src13 = GET_SRC(src, 13);
 const vec_t src14 = GET_SRC(src, 14);
 const vec_t src15 = GET_SRC(src, 15);
 dst = mad(wei0.s0, src0, dst);
 dst = mad(wei0.s1, src1, dst);
 dst = mad(wei0.s2, src2, dst);
 dst = mad(wei0.s3, src3, dst);
 dst = mad(wei0.s4, src4, dst);
 dst = mad(wei0.s5, src5, dst);
 dst = mad(wei0.s6, src6, dst);
 dst = mad(wei0.s7, src7, dst);
 dst = mad(wei1.s0, src8, dst);
 dst = mad(wei1.s1, src9, dst);
 dst = mad(wei1.s2, src10, dst);
 dst = mad(wei1.s3, src11, dst);
 dst = mad(wei1.s4, src12, dst);
 dst = mad(wei1.s5, src13, dst);
 dst = mad(wei1.s6, src14, dst);
 dst = mad(wei1.s7, src15, dst);
 }
#if SLM_DIV_FACTOR > 1
 partial_summ[lid1] = dst;
 barrier(CLK_LOCAL_MEM_FENCE);
 if (feature_sub_block == 0) {
 unroll_for(int i = 1; i < SLM_DIV_FACTOR; i++)
 dst += partial_summ[lid1 % feature_per_wg + i * feature_per_wg];
#endif
 dst = ACTIVATION(dst, ACTIVATION_PARAMS);
#if OUTPUT_LEFTOVERS
 if ((feature_block + 1) * FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM)
 {
#if X_BLOCK_SIZE > 1
 for (int i = 0; i < X_BLOCK_SIZE; i++) {
 if (xy * X_BLOCK_SIZE + i >= OUTPUT_SIZE_X * OUTPUT_SIZE_Y || feature_block * FEATURE_SLICE_SIZE + sglid >= OUTPUT_FEATURE_NUM)
 return;
 int xi = (x + i) % OUTPUT_SIZE_X;
 int yi = y + ((x + i) / OUTPUT_SIZE_X);
#if HAS_FUSED_OPS
 FUSED_OPS_SCALAR;
 dst[i] = FUSED_OPS_RESULT_SCALAR;
#endif
 output[output_offset + yi * output_y_pitch + xi * output_x_pitch + sglid] = dst[i];
 }
#else
 if (feature_block * FEATURE_SLICE_SIZE + sglid >= OUTPUT_FEATURE_NUM)
 return;
#if HAS_FUSED_OPS
 FUSED_OPS_SCALAR_B1;
 dst = FUSED_OPS_RESULT_SCALAR_B1;
#endif
 output[output_offset + sglid] = dst;
#endif
 }
 else
#endif
#if X_BLOCK_SIZE > 1
 {
#if !PADDED_OUTPUT && !NON_UNIT_FUSED_OP_SPATIAL
 if (xy * X_BLOCK_SIZE + X_BLOCK_SIZE <= OUTPUT_SIZE_X * OUTPUT_SIZE_Y || (OUTPUT_SIZE_X * OUTPUT_SIZE_Y) % X_BLOCK_SIZE == 0) {
#else
 if (x + X_BLOCK_SIZE <= OUTPUT_SIZE_X || OUTPUT_SIZE_X % X_BLOCK_SIZE == 0) {
#endif
#if HAS_FUSED_OPS
 FUSED_OPS_VEC;
 dst = FUSED_OPS_RESULT_VEC;
#endif
 UNIT_BLOCK_WRITE_VEC(output, output_offset + y * output_y_pitch + x * output_x_pitch, dst);
 } else {
 for (int i = 0; i < X_BLOCK_SIZE; i++) {
 if (xy * X_BLOCK_SIZE + i >= OUTPUT_SIZE_X * OUTPUT_SIZE_Y)
 return;
 int xi = (x + i) % OUTPUT_SIZE_X;
 int yi = y + ((x + i) / OUTPUT_SIZE_X);
#if HAS_FUSED_OPS
 FUSED_OPS_SCALAR;
 dst[i] = FUSED_OPS_RESULT_SCALAR;
#endif
 UNIT_BLOCK_WRITE(output, output_offset + yi * output_y_pitch + xi * output_x_pitch, dst[i]);
 }
 }
 }
#else
 {
#if HAS_FUSED_OPS
 FUSED_OPS_SCALAR_B1;
 dst = FUSED_OPS_RESULT_SCALAR_B1;
#endif
 UNIT_BLOCK_WRITE(output, output_offset, dst);
 }
#endif
#if SLM_DIV_FACTOR > 1
 }
#endif
}
#undef GET_SRC
#undef FEATURE_SLICE_SIZE
#undef UNIT_BLOCK_READ_VEC
#undef UNIT_BLOCK_WRITE_VEC
#ifdef FEATURE_SLICE_SIZE
#undef FEATURE_SLICE_SIZE
#endif
#ifdef UNIT_BLOCK_RW_TYPE
#undef UNIT_BLOCK_RW_TYPE
#endif
#ifdef UNIT_BLOCK_READ_FUNC
#undef UNIT_BLOCK_READ_FUNC
#endif
#ifdef UNIT_BLOCK_WRITE_FUNC
#undef UNIT_BLOCK_WRITE_FUNC
#endif
#ifdef UNIT_BLOCK_RW_TYPE
#undef UNIT_BLOCK_RW_TYPE
#endif
#ifdef UNIT_BLOCK_READ_FUNC
#undef UNIT_BLOCK_READ_FUNC
#endif
#ifdef UNIT_BLOCK_WRITE_FUNC
#undef UNIT_BLOCK_WRITE_FUNC
#endif
#ifdef UNIT_TYPE2
#undef UNIT_TYPE2
#endif
#ifdef UNIT_TYPE4
#undef UNIT_TYPE4
#endif
#ifdef UNIT_TYPE8
#undef UNIT_TYPE8
#endif
#ifdef UNIT_BLOCK_RW_TYPE2
#undef UNIT_BLOCK_RW_TYPE2
#endif
#ifdef UNIT_BLOCK_RW_TYPE4
#undef UNIT_BLOCK_RW_TYPE4
#endif
#ifdef UNIT_BLOCK_RW_TYPE8
#undef UNIT_BLOCK_RW_TYPE8
#endif
#ifdef UNIT_BLOCK_READ_FUNC2
#undef UNIT_BLOCK_READ_FUNC2
#endif
#ifdef UNIT_BLOCK_READ_FUNC4
#undef UNIT_BLOCK_READ_FUNC4
#endif
#ifdef UNIT_BLOCK_READ_FUNC8
#undef UNIT_BLOCK_READ_FUNC8
#endif
#ifdef UNIT_BLOCK_WRITE_FUNC2
#undef UNIT_BLOCK_WRITE_FUNC2
#endif
#ifdef UNIT_BLOCK_WRITE_FUNC4
#undef UNIT_BLOCK_WRITE_FUNC4
#endif
#ifdef UNIT_BLOCK_WRITE_FUNC8
#undef UNIT_BLOCK_WRITE_FUNC8
#endif
#ifdef UNIT_BLOCK_READ
#undef UNIT_BLOCK_READ
#endif
#ifdef UNIT_BLOCK_READ2
#undef UNIT_BLOCK_READ2
#endif
#ifdef UNIT_BLOCK_READ4
#undef UNIT_BLOCK_READ4
#endif
#ifdef UNIT_BLOCK_READ8
#undef UNIT_BLOCK_READ8
#endif
#ifdef UNIT_BLOCK_WRITE
#undef UNIT_BLOCK_WRITE
#endif
#ifdef UNIT_BLOCK_WRITE2
#undef UNIT_BLOCK_WRITE2
#endif
#ifdef UNIT_BLOCK_WRITE4
#undef UNIT_BLOCK_WRITE4
#endif
#ifdef UNIT_BLOCK_WRITE8
#undef UNIT_BLOCK_WRITE8
#endif
#undef KERNEL
#undef KERNEL_ID
#undef FUNC
#undef FUNC_CALL
#undef CONST_ARRAY_DECL
#undef CONST_ARRAY_REF
#ifdef FP64_SUPPORTED
#undef FP64_SUPPORTED
#endif
#ifdef FP16_SUPPORTED
#undef FP16_SUPPORTED
#endif
#ifdef FP16_UNIT_USED
#undef FP16_UNIT_USED
#endif
#ifdef INT8_UNIT_USED
#undef INT8_UNIT_USED
#endif
#ifdef INT32_UNIT_USED
#undef INT32_UNIT_USED
#endif
#ifdef INT64_UNIT_USED
#undef INT64_UNIT_USED
#endif
#ifdef UINT8_UNIT_USED
#undef UINT8_UNIT_USED
#endif
#ifdef UINT32_UNIT_USED
#undef UINT32_UNIT_USED
#endif
#ifdef UNIT_TYPE
#undef UNIT_TYPE
#endif
#ifdef UNIT_VAL_MAX
#undef UNIT_VAL_MAX
#endif
#ifdef UNIT_VAL_MIN
#undef UNIT_VAL_MIN
#endif
#ifdef UNIT_VAL_ONE
#undef UNIT_VAL_ONE
#endif
#ifdef UNIT_VAL_ZERO
#undef UNIT_VAL_ZERO
#endif
#ifdef TO_UNIT_TYPE
#undef TO_UNIT_TYPE
#endif
#ifdef TO_UNIT_TYPE_SAT
#undef TO_UNIT_TYPE_SAT
#endif
#ifdef AS_UNIT_TYPE
#undef AS_UNIT_TYPE
#endif
#ifdef UNIT_MAX_FUNC
#undef UNIT_MAX_FUNC
#endif
#ifdef UNIT_MIN_FUNC
#undef UNIT_MIN_FUNC
#endif
#ifdef UNIT_ABS_FUNC
#undef UNIT_ABS_FUNC
#endif
#ifdef UNIT_TYPE_SIZE
#undef UNIT_TYPE_SIZE
#endif
#ifdef UNIT_IS_FP
#undef UNIT_IS_FP
#endif
#ifdef NL_M
#undef NL_M
#endif
#ifdef NL_N
#undef NL_N
#endif
#ifdef ACTIVATION_FUNC_TYPE
#undef ACTIVATION_FUNC_TYPE
#endif
#ifdef ACTIVATION_FUNC_VAL_MAX
#undef ACTIVATION_FUNC_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_VAL_MIN
#undef ACTIVATION_FUNC_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_VAL_ONE
#undef ACTIVATION_FUNC_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_VAL_ZERO
#undef ACTIVATION_FUNC_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_TYPE
#undef TO_ACTIVATION_FUNC_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_TYPE_SAT
#undef TO_ACTIVATION_FUNC_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_TYPE
#undef AS_ACTIVATION_FUNC_TYPE
#endif
#ifdef ACTIVATION_FUNC_MAX_FUNC
#undef ACTIVATION_FUNC_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_MIN_FUNC
#undef ACTIVATION_FUNC_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_ABS_FUNC
#undef ACTIVATION_FUNC_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_TYPE_SIZE
#undef ACTIVATION_FUNC_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_IS_FP
#undef ACTIVATION_FUNC_IS_FP
#endif
#ifdef ACTIVATION_PARAMS
#undef ACTIVATION_PARAMS
#endif
#ifdef ACTIVATION_FUNC
#undef ACTIVATION_FUNC
#endif
#ifdef ACTIVATION
#undef ACTIVATION
#endif
#ifdef INPUT0_SIZE_X
#undef INPUT0_SIZE_X
#endif
#ifdef INPUT0_SIZE_Y
#undef INPUT0_SIZE_Y
#endif
#ifdef INPUT0_SIZE_Z
#undef INPUT0_SIZE_Z
#endif
#ifdef INPUT0_SIZE_W
#undef INPUT0_SIZE_W
#endif
#ifdef INPUT0_SIZE_U
#undef INPUT0_SIZE_U
#endif
#ifdef INPUT0_SIZE_V
#undef INPUT0_SIZE_V
#endif
#ifdef INPUT0_FEATURE_NUM
#undef INPUT0_FEATURE_NUM
#endif
#ifdef INPUT0_BATCH_NUM
#undef INPUT0_BATCH_NUM
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_X
#undef INPUT0_PAD_BEFORE_SIZE_X
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_Y
#undef INPUT0_PAD_BEFORE_SIZE_Y
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_Z
#undef INPUT0_PAD_BEFORE_SIZE_Z
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_W
#undef INPUT0_PAD_BEFORE_SIZE_W
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_U
#undef INPUT0_PAD_BEFORE_SIZE_U
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_V
#undef INPUT0_PAD_BEFORE_SIZE_V
#endif
#ifdef INPUT0_PAD_BEFORE_FEATURE_NUM
#undef INPUT0_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef INPUT0_PAD_BEFORE_BATCH_NUM
#undef INPUT0_PAD_BEFORE_BATCH_NUM
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_X
#undef INPUT0_PAD_AFTER_SIZE_X
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_Y
#undef INPUT0_PAD_AFTER_SIZE_Y
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_Z
#undef INPUT0_PAD_AFTER_SIZE_Z
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_W
#undef INPUT0_PAD_AFTER_SIZE_W
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_U
#undef INPUT0_PAD_AFTER_SIZE_U
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_V
#undef INPUT0_PAD_AFTER_SIZE_V
#endif
#ifdef INPUT0_PAD_AFTER_FEATURE_NUM
#undef INPUT0_PAD_AFTER_FEATURE_NUM
#endif
#ifdef INPUT0_PAD_AFTER_BATCH_NUM
#undef INPUT0_PAD_AFTER_BATCH_NUM
#endif
#ifdef INPUT0_LENGTH
#undef INPUT0_LENGTH
#endif
#ifdef INPUT0_X_PITCH
#undef INPUT0_X_PITCH
#endif
#ifdef INPUT0_Y_PITCH
#undef INPUT0_Y_PITCH
#endif
#ifdef INPUT0_Z_PITCH
#undef INPUT0_Z_PITCH
#endif
#ifdef INPUT0_W_PITCH
#undef INPUT0_W_PITCH
#endif
#ifdef INPUT0_U_PITCH
#undef INPUT0_U_PITCH
#endif
#ifdef INPUT0_V_PITCH
#undef INPUT0_V_PITCH
#endif
#ifdef INPUT0_FEATURE_PITCH
#undef INPUT0_FEATURE_PITCH
#endif
#ifdef INPUT0_BATCH_PITCH
#undef INPUT0_BATCH_PITCH
#endif
#ifdef INPUT0_GET_INDEX_SAFE
#undef INPUT0_GET_INDEX_SAFE
#endif
#ifdef INPUT0_GET_INDEX
#undef INPUT0_GET_INDEX
#endif
#ifdef INPUT0_GET_INDEX_RAW
#undef INPUT0_GET_INDEX_RAW
#endif
#ifdef INPUT0_VIEW_OFFSET
#undef INPUT0_VIEW_OFFSET
#endif
#ifdef INPUT0_DIMS
#undef INPUT0_DIMS
#endif
#ifdef INPUT0_SIMPLE
#undef INPUT0_SIMPLE
#endif
#ifdef INPUT0_GROUPED
#undef INPUT0_GROUPED
#endif
#ifdef INPUT0_LAYOUT_B_FS_YX_FSV16
#undef INPUT0_LAYOUT_B_FS_YX_FSV16
#endif
#ifdef INPUT0_TYPE
#undef INPUT0_TYPE
#endif
#ifdef INPUT0_VAL_MAX
#undef INPUT0_VAL_MAX
#endif
#ifdef INPUT0_VAL_MIN
#undef INPUT0_VAL_MIN
#endif
#ifdef INPUT0_VAL_ONE
#undef INPUT0_VAL_ONE
#endif
#ifdef INPUT0_VAL_ZERO
#undef INPUT0_VAL_ZERO
#endif
#ifdef TO_INPUT0_TYPE
#undef TO_INPUT0_TYPE
#endif
#ifdef TO_INPUT0_TYPE_SAT
#undef TO_INPUT0_TYPE_SAT
#endif
#ifdef AS_INPUT0_TYPE
#undef AS_INPUT0_TYPE
#endif
#ifdef INPUT0_MAX_FUNC
#undef INPUT0_MAX_FUNC
#endif
#ifdef INPUT0_MIN_FUNC
#undef INPUT0_MIN_FUNC
#endif
#ifdef INPUT0_ABS_FUNC
#undef INPUT0_ABS_FUNC
#endif
#ifdef INPUT0_TYPE_SIZE
#undef INPUT0_TYPE_SIZE
#endif
#ifdef INPUT0_IS_FP
#undef INPUT0_IS_FP
#endif
#ifdef INPUT0_OFFSET
#undef INPUT0_OFFSET
#endif
#ifdef INPUT0_SIZES_DATA
#undef INPUT0_SIZES_DATA
#endif
#ifdef INPUT0_PITCHES
#undef INPUT0_PITCHES
#endif
#ifdef INPUT0_PAD_BEFORE
#undef INPUT0_PAD_BEFORE
#endif
#ifdef INPUT0_PAD_AFTER
#undef INPUT0_PAD_AFTER
#endif
#ifdef OUTPUT_SIZE_X
#undef OUTPUT_SIZE_X
#endif
#ifdef OUTPUT_SIZE_Y
#undef OUTPUT_SIZE_Y
#endif
#ifdef OUTPUT_SIZE_Z
#undef OUTPUT_SIZE_Z
#endif
#ifdef OUTPUT_SIZE_W
#undef OUTPUT_SIZE_W
#endif
#ifdef OUTPUT_SIZE_U
#undef OUTPUT_SIZE_U
#endif
#ifdef OUTPUT_SIZE_V
#undef OUTPUT_SIZE_V
#endif
#ifdef OUTPUT_FEATURE_NUM
#undef OUTPUT_FEATURE_NUM
#endif
#ifdef OUTPUT_BATCH_NUM
#undef OUTPUT_BATCH_NUM
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_X
#undef OUTPUT_PAD_BEFORE_SIZE_X
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_Y
#undef OUTPUT_PAD_BEFORE_SIZE_Y
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_Z
#undef OUTPUT_PAD_BEFORE_SIZE_Z
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_W
#undef OUTPUT_PAD_BEFORE_SIZE_W
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_U
#undef OUTPUT_PAD_BEFORE_SIZE_U
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_V
#undef OUTPUT_PAD_BEFORE_SIZE_V
#endif
#ifdef OUTPUT_PAD_BEFORE_FEATURE_NUM
#undef OUTPUT_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef OUTPUT_PAD_BEFORE_BATCH_NUM
#undef OUTPUT_PAD_BEFORE_BATCH_NUM
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_X
#undef OUTPUT_PAD_AFTER_SIZE_X
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_Y
#undef OUTPUT_PAD_AFTER_SIZE_Y
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_Z
#undef OUTPUT_PAD_AFTER_SIZE_Z
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_W
#undef OUTPUT_PAD_AFTER_SIZE_W
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_U
#undef OUTPUT_PAD_AFTER_SIZE_U
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_V
#undef OUTPUT_PAD_AFTER_SIZE_V
#endif
#ifdef OUTPUT_PAD_AFTER_FEATURE_NUM
#undef OUTPUT_PAD_AFTER_FEATURE_NUM
#endif
#ifdef OUTPUT_PAD_AFTER_BATCH_NUM
#undef OUTPUT_PAD_AFTER_BATCH_NUM
#endif
#ifdef OUTPUT_LENGTH
#undef OUTPUT_LENGTH
#endif
#ifdef OUTPUT_X_PITCH
#undef OUTPUT_X_PITCH
#endif
#ifdef OUTPUT_Y_PITCH
#undef OUTPUT_Y_PITCH
#endif
#ifdef OUTPUT_Z_PITCH
#undef OUTPUT_Z_PITCH
#endif
#ifdef OUTPUT_W_PITCH
#undef OUTPUT_W_PITCH
#endif
#ifdef OUTPUT_U_PITCH
#undef OUTPUT_U_PITCH
#endif
#ifdef OUTPUT_V_PITCH
#undef OUTPUT_V_PITCH
#endif
#ifdef OUTPUT_FEATURE_PITCH
#undef OUTPUT_FEATURE_PITCH
#endif
#ifdef OUTPUT_BATCH_PITCH
#undef OUTPUT_BATCH_PITCH
#endif
#ifdef OUTPUT_GET_INDEX_SAFE
#undef OUTPUT_GET_INDEX_SAFE
#endif
#ifdef OUTPUT_GET_INDEX
#undef OUTPUT_GET_INDEX
#endif
#ifdef OUTPUT_GET_INDEX_RAW
#undef OUTPUT_GET_INDEX_RAW
#endif
#ifdef OUTPUT_VIEW_OFFSET
#undef OUTPUT_VIEW_OFFSET
#endif
#ifdef OUTPUT_DIMS
#undef OUTPUT_DIMS
#endif
#ifdef OUTPUT_SIMPLE
#undef OUTPUT_SIMPLE
#endif
#ifdef OUTPUT_GROUPED
#undef OUTPUT_GROUPED
#endif
#ifdef OUTPUT_LAYOUT_B_FS_YX_FSV16
#undef OUTPUT_LAYOUT_B_FS_YX_FSV16
#endif
#ifdef OUTPUT_TYPE
#undef OUTPUT_TYPE
#endif
#ifdef OUTPUT_VAL_MAX
#undef OUTPUT_VAL_MAX
#endif
#ifdef OUTPUT_VAL_MIN
#undef OUTPUT_VAL_MIN
#endif
#ifdef OUTPUT_VAL_ONE
#undef OUTPUT_VAL_ONE
#endif
#ifdef OUTPUT_VAL_ZERO
#undef OUTPUT_VAL_ZERO
#endif
#ifdef TO_OUTPUT_TYPE
#undef TO_OUTPUT_TYPE
#endif
#ifdef TO_OUTPUT_TYPE_SAT
#undef TO_OUTPUT_TYPE_SAT
#endif
#ifdef AS_OUTPUT_TYPE
#undef AS_OUTPUT_TYPE
#endif
#ifdef OUTPUT_MAX_FUNC
#undef OUTPUT_MAX_FUNC
#endif
#ifdef OUTPUT_MIN_FUNC
#undef OUTPUT_MIN_FUNC
#endif
#ifdef OUTPUT_ABS_FUNC
#undef OUTPUT_ABS_FUNC
#endif
#ifdef OUTPUT_TYPE_SIZE
#undef OUTPUT_TYPE_SIZE
#endif
#ifdef OUTPUT_IS_FP
#undef OUTPUT_IS_FP
#endif
#ifdef OUTPUT_OFFSET
#undef OUTPUT_OFFSET
#endif
#ifdef OUTPUT_SIZES_DATA
#undef OUTPUT_SIZES_DATA
#endif
#ifdef OUTPUT_PITCHES
#undef OUTPUT_PITCHES
#endif
#ifdef OUTPUT_PAD_BEFORE
#undef OUTPUT_PAD_BEFORE
#endif
#ifdef OUTPUT_PAD_AFTER
#undef OUTPUT_PAD_AFTER
#endif
#ifdef OPTIONAL_SHAPE_INFO_ARG
#undef OPTIONAL_SHAPE_INFO_ARG
#endif
#ifdef OPTIONAL_SHAPE_INFO_TENSOR
#undef OPTIONAL_SHAPE_INFO_TENSOR
#endif
#ifdef FILTER_SIZE_X
#undef FILTER_SIZE_X
#endif
#ifdef FILTER_SIZE_Y
#undef FILTER_SIZE_Y
#endif
#ifdef FILTER_SIZE_Z
#undef FILTER_SIZE_Z
#endif
#ifdef FILTER_IFM_NUM
#undef FILTER_IFM_NUM
#endif
#ifdef FILTER_OFM_NUM
#undef FILTER_OFM_NUM
#endif
#ifdef FILTER_GROUPS_NUM
#undef FILTER_GROUPS_NUM
#endif
#ifdef FILTER_X_PITCH
#undef FILTER_X_PITCH
#endif
#ifdef FILTER_Y_PITCH
#undef FILTER_Y_PITCH
#endif
#ifdef FILTER_Z_PITCH
#undef FILTER_Z_PITCH
#endif
#ifdef FILTER_IFM_PITCH
#undef FILTER_IFM_PITCH
#endif
#ifdef FILTER_OFM_PITCH
#undef FILTER_OFM_PITCH
#endif
#ifdef FILTER_GROUPS_PITCH
#undef FILTER_GROUPS_PITCH
#endif
#ifdef FILTER_LENGTH
#undef FILTER_LENGTH
#endif
#ifdef FILTER_VIEW_OFFSET
#undef FILTER_VIEW_OFFSET
#endif
#ifdef FILTER_DIMS
#undef FILTER_DIMS
#endif
#ifdef FILTER_SIMPLE
#undef FILTER_SIMPLE
#endif
#ifdef FILTER_GROUPED
#undef FILTER_GROUPED
#endif
#ifdef FILTER_LAYOUT_OS_IS_YX_ISV16_OSV16
#undef FILTER_LAYOUT_OS_IS_YX_ISV16_OSV16
#endif
#ifdef FILTER_TYPE
#undef FILTER_TYPE
#endif
#ifdef FILTER_VAL_MAX
#undef FILTER_VAL_MAX
#endif
#ifdef FILTER_VAL_MIN
#undef FILTER_VAL_MIN
#endif
#ifdef FILTER_VAL_ONE
#undef FILTER_VAL_ONE
#endif
#ifdef FILTER_VAL_ZERO
#undef FILTER_VAL_ZERO
#endif
#ifdef TO_FILTER_TYPE
#undef TO_FILTER_TYPE
#endif
#ifdef TO_FILTER_TYPE_SAT
#undef TO_FILTER_TYPE_SAT
#endif
#ifdef AS_FILTER_TYPE
#undef AS_FILTER_TYPE
#endif
#ifdef FILTER_MAX_FUNC
#undef FILTER_MAX_FUNC
#endif
#ifdef FILTER_MIN_FUNC
#undef FILTER_MIN_FUNC
#endif
#ifdef FILTER_ABS_FUNC
#undef FILTER_ABS_FUNC
#endif
#ifdef FILTER_TYPE_SIZE
#undef FILTER_TYPE_SIZE
#endif
#ifdef FILTER_IS_FP
#undef FILTER_IS_FP
#endif
#ifdef FILTER_OFFSET
#undef FILTER_OFFSET
#endif
#ifdef FILTER_SIZES_DATA
#undef FILTER_SIZES_DATA
#endif
#ifdef FILTER_PITCHES
#undef FILTER_PITCHES
#endif
#ifdef FILTER_PAD_BEFORE
#undef FILTER_PAD_BEFORE
#endif
#ifdef FILTER_PAD_AFTER
#undef FILTER_PAD_AFTER
#endif
#ifdef BIAS_TERM
#undef BIAS_TERM
#endif
#ifdef BIAS_SIZE_X
#undef BIAS_SIZE_X
#endif
#ifdef BIAS_SIZE_Y
#undef BIAS_SIZE_Y
#endif
#ifdef BIAS_SIZE_Z
#undef BIAS_SIZE_Z
#endif
#ifdef BIAS_SIZE_W
#undef BIAS_SIZE_W
#endif
#ifdef BIAS_SIZE_U
#undef BIAS_SIZE_U
#endif
#ifdef BIAS_SIZE_V
#undef BIAS_SIZE_V
#endif
#ifdef BIAS_FEATURE_NUM
#undef BIAS_FEATURE_NUM
#endif
#ifdef BIAS_BATCH_NUM
#undef BIAS_BATCH_NUM
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_X
#undef BIAS_PAD_BEFORE_SIZE_X
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_Y
#undef BIAS_PAD_BEFORE_SIZE_Y
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_Z
#undef BIAS_PAD_BEFORE_SIZE_Z
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_W
#undef BIAS_PAD_BEFORE_SIZE_W
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_U
#undef BIAS_PAD_BEFORE_SIZE_U
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_V
#undef BIAS_PAD_BEFORE_SIZE_V
#endif
#ifdef BIAS_PAD_BEFORE_FEATURE_NUM
#undef BIAS_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef BIAS_PAD_BEFORE_BATCH_NUM
#undef BIAS_PAD_BEFORE_BATCH_NUM
#endif
#ifdef BIAS_PAD_AFTER_SIZE_X
#undef BIAS_PAD_AFTER_SIZE_X
#endif
#ifdef BIAS_PAD_AFTER_SIZE_Y
#undef BIAS_PAD_AFTER_SIZE_Y
#endif
#ifdef BIAS_PAD_AFTER_SIZE_Z
#undef BIAS_PAD_AFTER_SIZE_Z
#endif
#ifdef BIAS_PAD_AFTER_SIZE_W
#undef BIAS_PAD_AFTER_SIZE_W
#endif
#ifdef BIAS_PAD_AFTER_SIZE_U
#undef BIAS_PAD_AFTER_SIZE_U
#endif
#ifdef BIAS_PAD_AFTER_SIZE_V
#undef BIAS_PAD_AFTER_SIZE_V
#endif
#ifdef BIAS_PAD_AFTER_FEATURE_NUM
#undef BIAS_PAD_AFTER_FEATURE_NUM
#endif
#ifdef BIAS_PAD_AFTER_BATCH_NUM
#undef BIAS_PAD_AFTER_BATCH_NUM
#endif
#ifdef BIAS_LENGTH
#undef BIAS_LENGTH
#endif
#ifdef BIAS_X_PITCH
#undef BIAS_X_PITCH
#endif
#ifdef BIAS_Y_PITCH
#undef BIAS_Y_PITCH
#endif
#ifdef BIAS_Z_PITCH
#undef BIAS_Z_PITCH
#endif
#ifdef BIAS_W_PITCH
#undef BIAS_W_PITCH
#endif
#ifdef BIAS_U_PITCH
#undef BIAS_U_PITCH
#endif
#ifdef BIAS_V_PITCH
#undef BIAS_V_PITCH
#endif
#ifdef BIAS_FEATURE_PITCH
#undef BIAS_FEATURE_PITCH
#endif
#ifdef BIAS_BATCH_PITCH
#undef BIAS_BATCH_PITCH
#endif
#ifdef BIAS_GET_INDEX_SAFE
#undef BIAS_GET_INDEX_SAFE
#endif
#ifdef BIAS_GET_INDEX
#undef BIAS_GET_INDEX
#endif
#ifdef BIAS_GET_INDEX_RAW
#undef BIAS_GET_INDEX_RAW
#endif
#ifdef BIAS_VIEW_OFFSET
#undef BIAS_VIEW_OFFSET
#endif
#ifdef BIAS_DIMS
#undef BIAS_DIMS
#endif
#ifdef BIAS_SIMPLE
#undef BIAS_SIMPLE
#endif
#ifdef BIAS_GROUPED
#undef BIAS_GROUPED
#endif
#ifdef BIAS_LAYOUT_BF
#undef BIAS_LAYOUT_BF
#endif
#ifdef BIAS_TYPE
#undef BIAS_TYPE
#endif
#ifdef BIAS_VAL_MAX
#undef BIAS_VAL_MAX
#endif
#ifdef BIAS_VAL_MIN
#undef BIAS_VAL_MIN
#endif
#ifdef BIAS_VAL_ONE
#undef BIAS_VAL_ONE
#endif
#ifdef BIAS_VAL_ZERO
#undef BIAS_VAL_ZERO
#endif
#ifdef TO_BIAS_TYPE
#undef TO_BIAS_TYPE
#endif
#ifdef TO_BIAS_TYPE_SAT
#undef TO_BIAS_TYPE_SAT
#endif
#ifdef AS_BIAS_TYPE
#undef AS_BIAS_TYPE
#endif
#ifdef BIAS_MAX_FUNC
#undef BIAS_MAX_FUNC
#endif
#ifdef BIAS_MIN_FUNC
#undef BIAS_MIN_FUNC
#endif
#ifdef BIAS_ABS_FUNC
#undef BIAS_ABS_FUNC
#endif
#ifdef BIAS_TYPE_SIZE
#undef BIAS_TYPE_SIZE
#endif
#ifdef BIAS_IS_FP
#undef BIAS_IS_FP
#endif
#ifdef BIAS_OFFSET
#undef BIAS_OFFSET
#endif
#ifdef BIAS_SIZES_DATA
#undef BIAS_SIZES_DATA
#endif
#ifdef BIAS_PITCHES
#undef BIAS_PITCHES
#endif
#ifdef BIAS_PAD_BEFORE
#undef BIAS_PAD_BEFORE
#endif
#ifdef BIAS_PAD_AFTER
#undef BIAS_PAD_AFTER
#endif
#ifdef BIAS_PER_OUTPUT
#undef BIAS_PER_OUTPUT
#endif
#ifdef BIAS_PER_OFM
#undef BIAS_PER_OFM
#endif
#ifdef STRIDE_SIZE_X
#undef STRIDE_SIZE_X
#endif
#ifdef STRIDE_SIZE_Y
#undef STRIDE_SIZE_Y
#endif
#ifdef STRIDE_SIZE_Z
#undef STRIDE_SIZE_Z
#endif
#ifdef PADDING_SIZE_X
#undef PADDING_SIZE_X
#endif
#ifdef PADDING_SIZE_Y
#undef PADDING_SIZE_Y
#endif
#ifdef PADDING_SIZE_Z
#undef PADDING_SIZE_Z
#endif
#ifdef DILATION_SIZE_X
#undef DILATION_SIZE_X
#endif
#ifdef DILATION_SIZE_Y
#undef DILATION_SIZE_Y
#endif
#ifdef DILATION_SIZE_Z
#undef DILATION_SIZE_Z
#endif
#ifdef FILTER_ARRAY_NUM
#undef FILTER_ARRAY_NUM
#endif
#ifdef INPUT0_OFFSET_WITH_PADDING
#undef INPUT0_OFFSET_WITH_PADDING
#endif
#ifdef GROUPED
#undef GROUPED
#endif
#ifdef SKIP_BATCH
#undef SKIP_BATCH
#endif
#ifdef FUSED_OP0_LOAD_VEC
#undef FUSED_OP0_LOAD_VEC
#endif
#ifdef FUSED_OP0_ACTION_VEC
#undef FUSED_OP0_ACTION_VEC
#endif
#ifdef FUSED_OP1_LOAD_VEC
#undef FUSED_OP1_LOAD_VEC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_MAX
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_MIN
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_ONE
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_ZERO
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE
#undef TO_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE_SAT
#undef TO_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE
#undef AS_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_MAX_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_MIN_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_ABS_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE_SIZE
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_IS_FP
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_IS_FP
#endif
#ifdef ACTIVATION_PARAMS_FUSED_OP1_VEC
#undef ACTIVATION_PARAMS_FUSED_OP1_VEC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC
#undef ACTIVATION_FUNC_FUSED_OP1_VEC
#endif
#ifdef ACTIVATION_FUSED_OP1_VEC
#undef ACTIVATION_FUSED_OP1_VEC
#endif
#ifdef FUSED_OP1_ACTION_VEC
#undef FUSED_OP1_ACTION_VEC
#endif
#ifdef FUSED_OPS_VEC
#undef FUSED_OPS_VEC
#endif
#ifdef FUSED_OPS_PRELOAD_VEC
#undef FUSED_OPS_PRELOAD_VEC
#endif
#ifdef FUSED_OPS_CALC_VEC
#undef FUSED_OPS_CALC_VEC
#endif
#ifdef FUSED_OPS_RESULT_VEC
#undef FUSED_OPS_RESULT_VEC
#endif
#ifdef FUSED_OPS_CAN_USE_PRELOAD_VEC
#undef FUSED_OPS_CAN_USE_PRELOAD_VEC
#endif
#ifdef FUSED_OP0_LOAD_SCALAR
#undef FUSED_OP0_LOAD_SCALAR
#endif
#ifdef FUSED_OP0_ACTION_SCALAR
#undef FUSED_OP0_ACTION_SCALAR
#endif
#ifdef FUSED_OP1_LOAD_SCALAR
#undef FUSED_OP1_LOAD_SCALAR
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_MAX
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_MIN
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_ONE
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_ZERO
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE
#undef TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE_SAT
#undef TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE
#undef AS_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_MAX_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_MIN_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_ABS_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE_SIZE
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_IS_FP
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_IS_FP
#endif
#ifdef ACTIVATION_PARAMS_FUSED_OP1_SCALAR
#undef ACTIVATION_PARAMS_FUSED_OP1_SCALAR
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR
#endif
#ifdef ACTIVATION_FUSED_OP1_SCALAR
#undef ACTIVATION_FUSED_OP1_SCALAR
#endif
#ifdef FUSED_OP1_ACTION_SCALAR
#undef FUSED_OP1_ACTION_SCALAR
#endif
#ifdef FUSED_OPS_SCALAR
#undef FUSED_OPS_SCALAR
#endif
#ifdef FUSED_OPS_PRELOAD_SCALAR
#undef FUSED_OPS_PRELOAD_SCALAR
#endif
#ifdef FUSED_OPS_CALC_SCALAR
#undef FUSED_OPS_CALC_SCALAR
#endif
#ifdef FUSED_OPS_RESULT_SCALAR
#undef FUSED_OPS_RESULT_SCALAR
#endif
#ifdef FUSED_OPS_CAN_USE_PRELOAD_SCALAR
#undef FUSED_OPS_CAN_USE_PRELOAD_SCALAR
#endif
#ifdef FUSED_OP0_LOAD_SCALAR_B1
#undef FUSED_OP0_LOAD_SCALAR_B1
#endif
#ifdef FUSED_OP0_ACTION_SCALAR_B1
#undef FUSED_OP0_ACTION_SCALAR_B1
#endif
#ifdef FUSED_OP1_LOAD_SCALAR_B1
#undef FUSED_OP1_LOAD_SCALAR_B1
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_MAX
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_MIN
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_ONE
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_ZERO
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE
#undef TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE_SAT
#undef TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE
#undef AS_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_MAX_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_MIN_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_ABS_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE_SIZE
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_IS_FP
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_IS_FP
#endif
#ifdef ACTIVATION_PARAMS_FUSED_OP1_SCALAR_B1
#undef ACTIVATION_PARAMS_FUSED_OP1_SCALAR_B1
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1
#endif
#ifdef ACTIVATION_FUSED_OP1_SCALAR_B1
#undef ACTIVATION_FUSED_OP1_SCALAR_B1
#endif
#ifdef FUSED_OP1_ACTION_SCALAR_B1
#undef FUSED_OP1_ACTION_SCALAR_B1
#endif
#ifdef FUSED_OPS_SCALAR_B1
#undef FUSED_OPS_SCALAR_B1
#endif
#ifdef FUSED_OPS_PRELOAD_SCALAR_B1
#undef FUSED_OPS_PRELOAD_SCALAR_B1
#endif
#ifdef FUSED_OPS_CALC_SCALAR_B1
#undef FUSED_OPS_CALC_SCALAR_B1
#endif
#ifdef FUSED_OPS_RESULT_SCALAR_B1
#undef FUSED_OPS_RESULT_SCALAR_B1
#endif
#ifdef FUSED_OPS_CAN_USE_PRELOAD_SCALAR_B1
#undef FUSED_OPS_CAN_USE_PRELOAD_SCALAR_B1
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_X
#undef FUSED_OP_0_INPUT0_SIZE_X
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_Y
#undef FUSED_OP_0_INPUT0_SIZE_Y
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_Z
#undef FUSED_OP_0_INPUT0_SIZE_Z
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_W
#undef FUSED_OP_0_INPUT0_SIZE_W
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_U
#undef FUSED_OP_0_INPUT0_SIZE_U
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_V
#undef FUSED_OP_0_INPUT0_SIZE_V
#endif
#ifdef FUSED_OP_0_INPUT0_FEATURE_NUM
#undef FUSED_OP_0_INPUT0_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_BATCH_NUM
#undef FUSED_OP_0_INPUT0_BATCH_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_X
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_X
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Y
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Y
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Z
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Z
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_W
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_W
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_U
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_U
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_V
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_V
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_FEATURE_NUM
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_BATCH_NUM
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_BATCH_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_X
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_X
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Y
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Y
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Z
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Z
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_W
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_W
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_U
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_U
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_V
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_V
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_FEATURE_NUM
#undef FUSED_OP_0_INPUT0_PAD_AFTER_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_BATCH_NUM
#undef FUSED_OP_0_INPUT0_PAD_AFTER_BATCH_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_LENGTH
#undef FUSED_OP_0_INPUT0_LENGTH
#endif
#ifdef FUSED_OP_0_INPUT0_X_PITCH
#undef FUSED_OP_0_INPUT0_X_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_Y_PITCH
#undef FUSED_OP_0_INPUT0_Y_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_Z_PITCH
#undef FUSED_OP_0_INPUT0_Z_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_W_PITCH
#undef FUSED_OP_0_INPUT0_W_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_U_PITCH
#undef FUSED_OP_0_INPUT0_U_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_V_PITCH
#undef FUSED_OP_0_INPUT0_V_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_FEATURE_PITCH
#undef FUSED_OP_0_INPUT0_FEATURE_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_BATCH_PITCH
#undef FUSED_OP_0_INPUT0_BATCH_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_GET_INDEX_SAFE
#undef FUSED_OP_0_INPUT0_GET_INDEX_SAFE
#endif
#ifdef FUSED_OP_0_INPUT0_GET_INDEX
#undef FUSED_OP_0_INPUT0_GET_INDEX
#endif
#ifdef FUSED_OP_0_INPUT0_GET_INDEX_RAW
#undef FUSED_OP_0_INPUT0_GET_INDEX_RAW
#endif
#ifdef FUSED_OP_0_INPUT0_VIEW_OFFSET
#undef FUSED_OP_0_INPUT0_VIEW_OFFSET
#endif
#ifdef FUSED_OP_0_INPUT0_DIMS
#undef FUSED_OP_0_INPUT0_DIMS
#endif
#ifdef FUSED_OP_0_INPUT0_SIMPLE
#undef FUSED_OP_0_INPUT0_SIMPLE
#endif
#ifdef FUSED_OP_0_INPUT0_GROUPED
#undef FUSED_OP_0_INPUT0_GROUPED
#endif
#ifdef FUSED_OP_0_INPUT0_LAYOUT_B_FS_YX_FSV16
#undef FUSED_OP_0_INPUT0_LAYOUT_B_FS_YX_FSV16
#endif
#ifdef FUSED_OP_0_INPUT0_TYPE
#undef FUSED_OP_0_INPUT0_TYPE
#endif
#ifdef FUSED_OP_0_INPUT0_VAL_MAX
#undef FUSED_OP_0_INPUT0_VAL_MAX
#endif
#ifdef FUSED_OP_0_INPUT0_VAL_MIN
#undef FUSED_OP_0_INPUT0_VAL_MIN
#endif
#ifdef FUSED_OP_0_INPUT0_VAL_ONE
#undef FUSED_OP_0_INPUT0_VAL_ONE
#endif
#ifdef FUSED_OP_0_INPUT0_VAL_ZERO
#undef FUSED_OP_0_INPUT0_VAL_ZERO
#endif
#ifdef TO_FUSED_OP_0_INPUT0_TYPE
#undef TO_FUSED_OP_0_INPUT0_TYPE
#endif
#ifdef TO_FUSED_OP_0_INPUT0_TYPE_SAT
#undef TO_FUSED_OP_0_INPUT0_TYPE_SAT
#endif
#ifdef AS_FUSED_OP_0_INPUT0_TYPE
#undef AS_FUSED_OP_0_INPUT0_TYPE
#endif
#ifdef FUSED_OP_0_INPUT0_MAX_FUNC
#undef FUSED_OP_0_INPUT0_MAX_FUNC
#endif
#ifdef FUSED_OP_0_INPUT0_MIN_FUNC
#undef FUSED_OP_0_INPUT0_MIN_FUNC
#endif
#ifdef FUSED_OP_0_INPUT0_ABS_FUNC
#undef FUSED_OP_0_INPUT0_ABS_FUNC
#endif
#ifdef FUSED_OP_0_INPUT0_TYPE_SIZE
#undef FUSED_OP_0_INPUT0_TYPE_SIZE
#endif
#ifdef FUSED_OP_0_INPUT0_IS_FP
#undef FUSED_OP_0_INPUT0_IS_FP
#endif
#ifdef FUSED_OP_0_INPUT0_OFFSET
#undef FUSED_OP_0_INPUT0_OFFSET
#endif
#ifdef FUSED_OP_0_INPUT0_SIZES_DATA
#undef FUSED_OP_0_INPUT0_SIZES_DATA
#endif
#ifdef FUSED_OP_0_INPUT0_PITCHES
#undef FUSED_OP_0_INPUT0_PITCHES
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE
#undef FUSED_OP_0_INPUT0_PAD_BEFORE
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER
#undef FUSED_OP_0_INPUT0_PAD_AFTER
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_X
#undef FUSED_OP_0_OUTPUT_SIZE_X
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_Y
#undef FUSED_OP_0_OUTPUT_SIZE_Y
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_Z
#undef FUSED_OP_0_OUTPUT_SIZE_Z
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_W
#undef FUSED_OP_0_OUTPUT_SIZE_W
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_U
#undef FUSED_OP_0_OUTPUT_SIZE_U
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_V
#undef FUSED_OP_0_OUTPUT_SIZE_V
#endif
#ifdef FUSED_OP_0_OUTPUT_FEATURE_NUM
#undef FUSED_OP_0_OUTPUT_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_BATCH_NUM
#undef FUSED_OP_0_OUTPUT_BATCH_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_X
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_X
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Y
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Y
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Z
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Z
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_W
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_W
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_U
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_U
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_V
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_V
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_FEATURE_NUM
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_BATCH_NUM
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_BATCH_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_X
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_X
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Y
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Y
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Z
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Z
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_W
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_W
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_U
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_U
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_V
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_V
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_FEATURE_NUM
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_BATCH_NUM
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_BATCH_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_LENGTH
#undef FUSED_OP_0_OUTPUT_LENGTH
#endif
#ifdef FUSED_OP_0_OUTPUT_X_PITCH
#undef FUSED_OP_0_OUTPUT_X_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_Y_PITCH
#undef FUSED_OP_0_OUTPUT_Y_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_Z_PITCH
#undef FUSED_OP_0_OUTPUT_Z_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_W_PITCH
#undef FUSED_OP_0_OUTPUT_W_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_U_PITCH
#undef FUSED_OP_0_OUTPUT_U_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_V_PITCH
#undef FUSED_OP_0_OUTPUT_V_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_FEATURE_PITCH
#undef FUSED_OP_0_OUTPUT_FEATURE_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_BATCH_PITCH
#undef FUSED_OP_0_OUTPUT_BATCH_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_GET_INDEX_SAFE
#undef FUSED_OP_0_OUTPUT_GET_INDEX_SAFE
#endif
#ifdef FUSED_OP_0_OUTPUT_GET_INDEX
#undef FUSED_OP_0_OUTPUT_GET_INDEX
#endif
#ifdef FUSED_OP_0_OUTPUT_GET_INDEX_RAW
#undef FUSED_OP_0_OUTPUT_GET_INDEX_RAW
#endif
#ifdef FUSED_OP_0_OUTPUT_VIEW_OFFSET
#undef FUSED_OP_0_OUTPUT_VIEW_OFFSET
#endif
#ifdef FUSED_OP_0_OUTPUT_DIMS
#undef FUSED_OP_0_OUTPUT_DIMS
#endif
#ifdef FUSED_OP_0_OUTPUT_SIMPLE
#undef FUSED_OP_0_OUTPUT_SIMPLE
#endif
#ifdef FUSED_OP_0_OUTPUT_GROUPED
#undef FUSED_OP_0_OUTPUT_GROUPED
#endif
#ifdef FUSED_OP_0_OUTPUT_LAYOUT_BFYX
#undef FUSED_OP_0_OUTPUT_LAYOUT_BFYX
#endif
#ifdef FUSED_OP_0_OUTPUT_TYPE
#undef FUSED_OP_0_OUTPUT_TYPE
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_MAX
#undef FUSED_OP_0_OUTPUT_VAL_MAX
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_MIN
#undef FUSED_OP_0_OUTPUT_VAL_MIN
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_ONE
#undef FUSED_OP_0_OUTPUT_VAL_ONE
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_ZERO
#undef FUSED_OP_0_OUTPUT_VAL_ZERO
#endif
#ifdef TO_FUSED_OP_0_OUTPUT_TYPE
#undef TO_FUSED_OP_0_OUTPUT_TYPE
#endif
#ifdef TO_FUSED_OP_0_OUTPUT_TYPE_SAT
#undef TO_FUSED_OP_0_OUTPUT_TYPE_SAT
#endif
#ifdef AS_FUSED_OP_0_OUTPUT_TYPE
#undef AS_FUSED_OP_0_OUTPUT_TYPE
#endif
#ifdef FUSED_OP_0_OUTPUT_MAX_FUNC
#undef FUSED_OP_0_OUTPUT_MAX_FUNC
#endif
#ifdef FUSED_OP_0_OUTPUT_MIN_FUNC
#undef FUSED_OP_0_OUTPUT_MIN_FUNC
#endif
#ifdef FUSED_OP_0_OUTPUT_ABS_FUNC
#undef FUSED_OP_0_OUTPUT_ABS_FUNC
#endif
#ifdef FUSED_OP_0_OUTPUT_TYPE_SIZE
#undef FUSED_OP_0_OUTPUT_TYPE_SIZE
#endif
#ifdef FUSED_OP_0_OUTPUT_IS_FP
#undef FUSED_OP_0_OUTPUT_IS_FP
#endif
#ifdef FUSED_OP_0_OUTPUT_OFFSET
#undef FUSED_OP_0_OUTPUT_OFFSET
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZES_DATA
#undef FUSED_OP_0_OUTPUT_SIZES_DATA
#endif
#ifdef FUSED_OP_0_OUTPUT_PITCHES
#undef FUSED_OP_0_OUTPUT_PITCHES
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER
#undef FUSED_OP_0_OUTPUT_PAD_AFTER
#endif
#ifdef FUSED_OP0_DECLS
#undef FUSED_OP0_DECLS
#endif
#ifdef FUSED_OP0_ARGS
#undef FUSED_OP0_ARGS
#endif
#ifdef FUSED_OP_1_OUTPUT_SIZE_X
#undef FUSED_OP_1_OUTPUT_SIZE_X
#endif
#ifdef FUSED_OP_1_OUTPUT_SIZE_Y
#undef FUSED_OP_1_OUTPUT_SIZE_Y
#endif
#ifdef FUSED_OP_1_OUTPUT_SIZE_Z
#undef FUSED_OP_1_OUTPUT_SIZE_Z
#endif
#ifdef FUSED_OP_1_OUTPUT_SIZE_W
#undef FUSED_OP_1_OUTPUT_SIZE_W
#endif
#ifdef FUSED_OP_1_OUTPUT_SIZE_U
#undef FUSED_OP_1_OUTPUT_SIZE_U
#endif
#ifdef FUSED_OP_1_OUTPUT_SIZE_V
#undef FUSED_OP_1_OUTPUT_SIZE_V
#endif
#ifdef FUSED_OP_1_OUTPUT_FEATURE_NUM
#undef FUSED_OP_1_OUTPUT_FEATURE_NUM
#endif
#ifdef FUSED_OP_1_OUTPUT_BATCH_NUM
#undef FUSED_OP_1_OUTPUT_BATCH_NUM
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_X
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_X
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_Y
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_Y
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_Z
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_Z
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_W
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_W
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_U
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_U
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_V
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_V
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE_FEATURE_NUM
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE_BATCH_NUM
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE_BATCH_NUM
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_X
#undef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_X
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_Y
#undef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_Y
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_Z
#undef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_Z
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_W
#undef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_W
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_U
#undef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_U
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_V
#undef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_V
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER_FEATURE_NUM
#undef FUSED_OP_1_OUTPUT_PAD_AFTER_FEATURE_NUM
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER_BATCH_NUM
#undef FUSED_OP_1_OUTPUT_PAD_AFTER_BATCH_NUM
#endif
#ifdef FUSED_OP_1_OUTPUT_LENGTH
#undef FUSED_OP_1_OUTPUT_LENGTH
#endif
#ifdef FUSED_OP_1_OUTPUT_X_PITCH
#undef FUSED_OP_1_OUTPUT_X_PITCH
#endif
#ifdef FUSED_OP_1_OUTPUT_Y_PITCH
#undef FUSED_OP_1_OUTPUT_Y_PITCH
#endif
#ifdef FUSED_OP_1_OUTPUT_Z_PITCH
#undef FUSED_OP_1_OUTPUT_Z_PITCH
#endif
#ifdef FUSED_OP_1_OUTPUT_W_PITCH
#undef FUSED_OP_1_OUTPUT_W_PITCH
#endif
#ifdef FUSED_OP_1_OUTPUT_U_PITCH
#undef FUSED_OP_1_OUTPUT_U_PITCH
#endif
#ifdef FUSED_OP_1_OUTPUT_V_PITCH
#undef FUSED_OP_1_OUTPUT_V_PITCH
#endif
#ifdef FUSED_OP_1_OUTPUT_FEATURE_PITCH
#undef FUSED_OP_1_OUTPUT_FEATURE_PITCH
#endif
#ifdef FUSED_OP_1_OUTPUT_BATCH_PITCH
#undef FUSED_OP_1_OUTPUT_BATCH_PITCH
#endif
#ifdef FUSED_OP_1_OUTPUT_GET_INDEX_SAFE
#undef FUSED_OP_1_OUTPUT_GET_INDEX_SAFE
#endif
#ifdef FUSED_OP_1_OUTPUT_GET_INDEX
#undef FUSED_OP_1_OUTPUT_GET_INDEX
#endif
#ifdef FUSED_OP_1_OUTPUT_GET_INDEX_RAW
#undef FUSED_OP_1_OUTPUT_GET_INDEX_RAW
#endif
#ifdef FUSED_OP_1_OUTPUT_VIEW_OFFSET
#undef FUSED_OP_1_OUTPUT_VIEW_OFFSET
#endif
#ifdef FUSED_OP_1_OUTPUT_DIMS
#undef FUSED_OP_1_OUTPUT_DIMS
#endif
#ifdef FUSED_OP_1_OUTPUT_SIMPLE
#undef FUSED_OP_1_OUTPUT_SIMPLE
#endif
#ifdef FUSED_OP_1_OUTPUT_GROUPED
#undef FUSED_OP_1_OUTPUT_GROUPED
#endif
#ifdef FUSED_OP_1_OUTPUT_LAYOUT_BFYX
#undef FUSED_OP_1_OUTPUT_LAYOUT_BFYX
#endif
#ifdef FUSED_OP_1_OUTPUT_TYPE
#undef FUSED_OP_1_OUTPUT_TYPE
#endif
#ifdef FUSED_OP_1_OUTPUT_VAL_MAX
#undef FUSED_OP_1_OUTPUT_VAL_MAX
#endif
#ifdef FUSED_OP_1_OUTPUT_VAL_MIN
#undef FUSED_OP_1_OUTPUT_VAL_MIN
#endif
#ifdef FUSED_OP_1_OUTPUT_VAL_ONE
#undef FUSED_OP_1_OUTPUT_VAL_ONE
#endif
#ifdef FUSED_OP_1_OUTPUT_VAL_ZERO
#undef FUSED_OP_1_OUTPUT_VAL_ZERO
#endif
#ifdef TO_FUSED_OP_1_OUTPUT_TYPE
#undef TO_FUSED_OP_1_OUTPUT_TYPE
#endif
#ifdef TO_FUSED_OP_1_OUTPUT_TYPE_SAT
#undef TO_FUSED_OP_1_OUTPUT_TYPE_SAT
#endif
#ifdef AS_FUSED_OP_1_OUTPUT_TYPE
#undef AS_FUSED_OP_1_OUTPUT_TYPE
#endif
#ifdef FUSED_OP_1_OUTPUT_MAX_FUNC
#undef FUSED_OP_1_OUTPUT_MAX_FUNC
#endif
#ifdef FUSED_OP_1_OUTPUT_MIN_FUNC
#undef FUSED_OP_1_OUTPUT_MIN_FUNC
#endif
#ifdef FUSED_OP_1_OUTPUT_ABS_FUNC
#undef FUSED_OP_1_OUTPUT_ABS_FUNC
#endif
#ifdef FUSED_OP_1_OUTPUT_TYPE_SIZE
#undef FUSED_OP_1_OUTPUT_TYPE_SIZE
#endif
#ifdef FUSED_OP_1_OUTPUT_IS_FP
#undef FUSED_OP_1_OUTPUT_IS_FP
#endif
#ifdef FUSED_OP_1_OUTPUT_OFFSET
#undef FUSED_OP_1_OUTPUT_OFFSET
#endif
#ifdef FUSED_OP_1_OUTPUT_SIZES_DATA
#undef FUSED_OP_1_OUTPUT_SIZES_DATA
#endif
#ifdef FUSED_OP_1_OUTPUT_PITCHES
#undef FUSED_OP_1_OUTPUT_PITCHES
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER
#undef FUSED_OP_1_OUTPUT_PAD_AFTER
#endif
#ifdef FUSED_OP1_DECLS
#undef FUSED_OP1_DECLS
#endif
#ifdef FUSED_OP1_ARGS
#undef FUSED_OP1_ARGS
#endif
#ifdef FUSED_OPS_DECLS
#undef FUSED_OPS_DECLS
#endif
#ifdef FUSED_OPS_ARGS
#undef FUSED_OPS_ARGS
#endif
#ifdef HAS_FUSED_OPS
#undef HAS_FUSED_OPS
#endif
#ifdef HAS_FUSED_OPS_DECLS
#undef HAS_FUSED_OPS_DECLS
#endif
#ifdef X_BLOCK_SIZE
#undef X_BLOCK_SIZE
#endif
#ifdef SLM_DIV_FACTOR
#undef SLM_DIV_FACTOR
#endif
#ifdef SUB_GROUP_SIZE
#undef SUB_GROUP_SIZE
#endif
#ifdef WORK_GROUP_SIZE
#undef WORK_GROUP_SIZE
#endif
#ifdef PADDED_INPUT
#undef PADDED_INPUT
#endif
#ifdef PADDED_OUTPUT
#undef PADDED_OUTPUT
#endif
#ifdef NON_UNIT_FUSED_OP_SPATIAL
#undef NON_UNIT_FUSED_OP_SPATIAL
#endif
#ifdef IC_BLOCKS
#undef IC_BLOCKS
#endif
#ifdef INPUT_LEFTOVERS
#undef INPUT_LEFTOVERS
#endif

//====================================================
// Kernel template: convolution_gpu_bfyx_f16_1x1 
// Kernel name: convolution_gpu_bfyx_f16_1x1_547689880436035411_0_0
#define KERNEL(name) __kernel void convolution_gpu_bfyx_f16_1x1_547689880436035411_0_0
#define KERNEL_ID convolution_gpu_bfyx_f16_1x1_547689880436035411_0_0
#define FUNC(name)  _##name##_convolution_gpu_bfyx_f16_1x1_547689880436035411_0_0
#define FUNC_CALL(name)  _##name##_convolution_gpu_bfyx_f16_1x1_547689880436035411_0_0
#define CONST_ARRAY_DECL(name) __constant size_t  _##name##_convolution_gpu_bfyx_f16_1x1_547689880436035411_0_0 []
#define CONST_ARRAY_REF(name)  _##name##_convolution_gpu_bfyx_f16_1x1_547689880436035411_0_0
#define FP64_SUPPORTED 0
#define FP16_SUPPORTED 1
#define FP16_UNIT_USED 1
#define INT8_UNIT_USED 0
#define INT32_UNIT_USED 0
#define INT64_UNIT_USED 0
#define UINT8_UNIT_USED 0
#define UINT32_UNIT_USED 0
#define UNIT_TYPE half
#define UNIT_VAL_MAX HALF_MAX
#define UNIT_VAL_MIN -UNIT_VAL_MAX
#define UNIT_VAL_ONE 1.0h
#define UNIT_VAL_ZERO 0.0h
#define TO_UNIT_TYPE(v) convert_half(v)
#define TO_UNIT_TYPE_SAT(v) convert_half(v)
#define AS_UNIT_TYPE(v) as_half(v)
#define UNIT_MAX_FUNC fmax
#define UNIT_MIN_FUNC fmin
#define UNIT_ABS_FUNC fabs
#define UNIT_TYPE_SIZE 2
#define UNIT_IS_FP 1
#define NL_M as_float(0x0)/*0.000000e+00*/
#define NL_N as_float(0x0)/*0.000000e+00*/
#define ACTIVATION_FUNC_TYPE half
#define ACTIVATION_FUNC_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_VAL_MIN -ACTIVATION_FUNC_VAL_MAX
#define ACTIVATION_FUNC_VAL_ONE 1.0h
#define ACTIVATION_FUNC_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_MAX_FUNC fmax
#define ACTIVATION_FUNC_MIN_FUNC fmin
#define ACTIVATION_FUNC_ABS_FUNC fabs
#define ACTIVATION_FUNC_TYPE_SIZE 2
#define ACTIVATION_FUNC_IS_FP 1
#define ACTIVATION_PARAMS NL_M, NL_N
#define ACTIVATION_FUNC(input, m, n) input
#define ACTIVATION(input, params) ACTIVATION_FUNC(input, params)
#define INPUT0_SIZE_X 14
#define INPUT0_SIZE_Y 14
#define INPUT0_SIZE_Z 1
#define INPUT0_SIZE_W 1
#define INPUT0_SIZE_U 1
#define INPUT0_SIZE_V 1
#define INPUT0_FEATURE_NUM 256
#define INPUT0_BATCH_NUM 1
#define INPUT0_PAD_BEFORE_SIZE_X 0
#define INPUT0_PAD_BEFORE_SIZE_Y 0
#define INPUT0_PAD_BEFORE_SIZE_Z 0
#define INPUT0_PAD_BEFORE_SIZE_W 0
#define INPUT0_PAD_BEFORE_SIZE_U 0
#define INPUT0_PAD_BEFORE_SIZE_V 0
#define INPUT0_PAD_BEFORE_FEATURE_NUM 0
#define INPUT0_PAD_BEFORE_BATCH_NUM 0
#define INPUT0_PAD_AFTER_SIZE_X 0
#define INPUT0_PAD_AFTER_SIZE_Y 0
#define INPUT0_PAD_AFTER_SIZE_Z 0
#define INPUT0_PAD_AFTER_SIZE_W 0
#define INPUT0_PAD_AFTER_SIZE_U 0
#define INPUT0_PAD_AFTER_SIZE_V 0
#define INPUT0_PAD_AFTER_FEATURE_NUM 0
#define INPUT0_PAD_AFTER_BATCH_NUM 0
#define INPUT0_LENGTH 50176
#define INPUT0_X_PITCH 1
#define INPUT0_Y_PITCH 14
#define INPUT0_Z_PITCH 1
#define INPUT0_W_PITCH 1
#define INPUT0_U_PITCH 1
#define INPUT0_V_PITCH 1
#define INPUT0_FEATURE_PITCH 196
#define INPUT0_BATCH_PITCH 50176
#define INPUT0_GET_INDEX_SAFE(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX_SAFE(INPUT0, b, f, y, x)
#define INPUT0_GET_INDEX(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(INPUT0, b, f, y, x)
#define INPUT0_GET_INDEX_RAW(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(INPUT0, b, f, y, x)
#define INPUT0_VIEW_OFFSET 0
#define INPUT0_DIMS 4
#define INPUT0_SIMPLE 0
#define INPUT0_GROUPED 0
#define INPUT0_LAYOUT_B_FS_YX_FSV16 1
#define INPUT0_TYPE half
#define INPUT0_VAL_MAX HALF_MAX
#define INPUT0_VAL_MIN -INPUT0_VAL_MAX
#define INPUT0_VAL_ONE 1.0h
#define INPUT0_VAL_ZERO 0.0h
#define TO_INPUT0_TYPE(v) convert_half(v)
#define TO_INPUT0_TYPE_SAT(v) convert_half(v)
#define AS_INPUT0_TYPE(v) as_half(v)
#define INPUT0_MAX_FUNC fmax
#define INPUT0_MIN_FUNC fmin
#define INPUT0_ABS_FUNC fabs
#define INPUT0_TYPE_SIZE 2
#define INPUT0_IS_FP 1
#define INPUT0_OFFSET 0
#define INPUT0_SIZES_DATA { 14,14,256,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(INPUT0_SIZES) = INPUT0_SIZES_DATA;
#define INPUT0_SIZES CONST_ARRAY_REF(INPUT0_SIZES)
#define INPUT0_PITCHES (size_t []){ 1,14,196,50176,1,1,1,1,1, } 
#define INPUT0_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define INPUT0_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OUTPUT_SIZE_X 14
#define OUTPUT_SIZE_Y 14
#define OUTPUT_SIZE_Z 1
#define OUTPUT_SIZE_W 1
#define OUTPUT_SIZE_U 1
#define OUTPUT_SIZE_V 1
#define OUTPUT_FEATURE_NUM 1024
#define OUTPUT_BATCH_NUM 1
#define OUTPUT_PAD_BEFORE_SIZE_X 0
#define OUTPUT_PAD_BEFORE_SIZE_Y 0
#define OUTPUT_PAD_BEFORE_SIZE_Z 0
#define OUTPUT_PAD_BEFORE_SIZE_W 0
#define OUTPUT_PAD_BEFORE_SIZE_U 0
#define OUTPUT_PAD_BEFORE_SIZE_V 0
#define OUTPUT_PAD_BEFORE_FEATURE_NUM 0
#define OUTPUT_PAD_BEFORE_BATCH_NUM 0
#define OUTPUT_PAD_AFTER_SIZE_X 0
#define OUTPUT_PAD_AFTER_SIZE_Y 0
#define OUTPUT_PAD_AFTER_SIZE_Z 0
#define OUTPUT_PAD_AFTER_SIZE_W 0
#define OUTPUT_PAD_AFTER_SIZE_U 0
#define OUTPUT_PAD_AFTER_SIZE_V 0
#define OUTPUT_PAD_AFTER_FEATURE_NUM 0
#define OUTPUT_PAD_AFTER_BATCH_NUM 0
#define OUTPUT_LENGTH 200704
#define OUTPUT_X_PITCH 1
#define OUTPUT_Y_PITCH 14
#define OUTPUT_Z_PITCH 1
#define OUTPUT_W_PITCH 1
#define OUTPUT_U_PITCH 1
#define OUTPUT_V_PITCH 1
#define OUTPUT_FEATURE_PITCH 196
#define OUTPUT_BATCH_PITCH 200704
#define OUTPUT_GET_INDEX_SAFE(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX_SAFE(OUTPUT, b, f, y, x)
#define OUTPUT_GET_INDEX(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(OUTPUT, b, f, y, x)
#define OUTPUT_GET_INDEX_RAW(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(OUTPUT, b, f, y, x)
#define OUTPUT_VIEW_OFFSET 0
#define OUTPUT_DIMS 4
#define OUTPUT_SIMPLE 0
#define OUTPUT_GROUPED 0
#define OUTPUT_LAYOUT_B_FS_YX_FSV16 1
#define OUTPUT_TYPE half
#define OUTPUT_VAL_MAX HALF_MAX
#define OUTPUT_VAL_MIN -OUTPUT_VAL_MAX
#define OUTPUT_VAL_ONE 1.0h
#define OUTPUT_VAL_ZERO 0.0h
#define TO_OUTPUT_TYPE(v) convert_half(v)
#define TO_OUTPUT_TYPE_SAT(v) convert_half(v)
#define AS_OUTPUT_TYPE(v) as_half(v)
#define OUTPUT_MAX_FUNC fmax
#define OUTPUT_MIN_FUNC fmin
#define OUTPUT_ABS_FUNC fabs
#define OUTPUT_TYPE_SIZE 2
#define OUTPUT_IS_FP 1
#define OUTPUT_OFFSET 0
#define OUTPUT_SIZES_DATA { 14,14,1024,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(OUTPUT_SIZES) = OUTPUT_SIZES_DATA;
#define OUTPUT_SIZES CONST_ARRAY_REF(OUTPUT_SIZES)
#define OUTPUT_PITCHES (size_t []){ 1,14,196,200704,1,1,1,1,1, } 
#define OUTPUT_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OUTPUT_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define OPTIONAL_SHAPE_INFO_ARG 
#define OPTIONAL_SHAPE_INFO_TENSOR 
#define FILTER_SIZE_X 1
#define FILTER_SIZE_Y 1
#define FILTER_SIZE_Z 1
#define FILTER_IFM_NUM 256
#define FILTER_OFM_NUM 1024
#define FILTER_GROUPS_NUM 1
#define FILTER_X_PITCH 1
#define FILTER_Y_PITCH 1
#define FILTER_Z_PITCH 1
#define FILTER_IFM_PITCH 1
#define FILTER_OFM_PITCH 256
#define FILTER_GROUPS_PITCH 1
#define FILTER_LENGTH 262144
#define FILTER_VIEW_OFFSET 0
#define FILTER_DIMS 4
#define FILTER_SIMPLE 0
#define FILTER_GROUPED 0
#define FILTER_LAYOUT_OS_IS_YX_ISV16_OSV16 1
#define FILTER_TYPE half
#define FILTER_VAL_MAX HALF_MAX
#define FILTER_VAL_MIN -FILTER_VAL_MAX
#define FILTER_VAL_ONE 1.0h
#define FILTER_VAL_ZERO 0.0h
#define TO_FILTER_TYPE(v) convert_half(v)
#define TO_FILTER_TYPE_SAT(v) convert_half(v)
#define AS_FILTER_TYPE(v) as_half(v)
#define FILTER_MAX_FUNC fmax
#define FILTER_MIN_FUNC fmin
#define FILTER_ABS_FUNC fabs
#define FILTER_TYPE_SIZE 2
#define FILTER_IS_FP 1
#define FILTER_OFFSET 0
#define FILTER_SIZES_DATA { 1,1,256,1024,1,1,1,1,1, } 
CONST_ARRAY_DECL(FILTER_SIZES) = FILTER_SIZES_DATA;
#define FILTER_SIZES CONST_ARRAY_REF(FILTER_SIZES)
#define FILTER_PITCHES (size_t []){ 1,1,1,256,1,1,1,1,1, } 
#define FILTER_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FILTER_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define BIAS_TERM 1
#define BIAS_SIZE_X 1
#define BIAS_SIZE_Y 1
#define BIAS_SIZE_Z 1
#define BIAS_SIZE_W 1
#define BIAS_SIZE_U 1
#define BIAS_SIZE_V 1
#define BIAS_FEATURE_NUM 1024
#define BIAS_BATCH_NUM 1
#define BIAS_PAD_BEFORE_SIZE_X 0
#define BIAS_PAD_BEFORE_SIZE_Y 0
#define BIAS_PAD_BEFORE_SIZE_Z 0
#define BIAS_PAD_BEFORE_SIZE_W 0
#define BIAS_PAD_BEFORE_SIZE_U 0
#define BIAS_PAD_BEFORE_SIZE_V 0
#define BIAS_PAD_BEFORE_FEATURE_NUM 0
#define BIAS_PAD_BEFORE_BATCH_NUM 0
#define BIAS_PAD_AFTER_SIZE_X 0
#define BIAS_PAD_AFTER_SIZE_Y 0
#define BIAS_PAD_AFTER_SIZE_Z 0
#define BIAS_PAD_AFTER_SIZE_W 0
#define BIAS_PAD_AFTER_SIZE_U 0
#define BIAS_PAD_AFTER_SIZE_V 0
#define BIAS_PAD_AFTER_FEATURE_NUM 0
#define BIAS_PAD_AFTER_BATCH_NUM 0
#define BIAS_LENGTH 1024
#define BIAS_X_PITCH 1
#define BIAS_Y_PITCH 1
#define BIAS_Z_PITCH 1
#define BIAS_W_PITCH 1
#define BIAS_U_PITCH 1
#define BIAS_V_PITCH 1
#define BIAS_FEATURE_PITCH 1
#define BIAS_BATCH_PITCH 1024
#define BIAS_GET_INDEX_SAFE(b, f, y, x) ((0 + (f)) % 1024)
#define BIAS_GET_INDEX(b, f, y, x) (0 + (f))
#define BIAS_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(BIAS, b, f, y, x)
#define BIAS_VIEW_OFFSET 0
#define BIAS_DIMS 2
#define BIAS_SIMPLE 1
#define BIAS_GROUPED 0
#define BIAS_LAYOUT_BF 1
#define BIAS_TYPE half
#define BIAS_VAL_MAX HALF_MAX
#define BIAS_VAL_MIN -BIAS_VAL_MAX
#define BIAS_VAL_ONE 1.0h
#define BIAS_VAL_ZERO 0.0h
#define TO_BIAS_TYPE(v) convert_half(v)
#define TO_BIAS_TYPE_SAT(v) convert_half(v)
#define AS_BIAS_TYPE(v) as_half(v)
#define BIAS_MAX_FUNC fmax
#define BIAS_MIN_FUNC fmin
#define BIAS_ABS_FUNC fabs
#define BIAS_TYPE_SIZE 2
#define BIAS_IS_FP 1
#define BIAS_OFFSET 0
#define BIAS_SIZES_DATA { 1024,1,1,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(BIAS_SIZES) = BIAS_SIZES_DATA;
#define BIAS_SIZES CONST_ARRAY_REF(BIAS_SIZES)
#define BIAS_PITCHES (size_t []){ 1,1024,1,1,1,1,1,1,1, } 
#define BIAS_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define BIAS_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define BIAS_PER_OUTPUT 0
#define BIAS_PER_OFM 1
#define STRIDE_SIZE_X 1
#define STRIDE_SIZE_Y 1
#define STRIDE_SIZE_Z 1
#define PADDING_SIZE_X 0
#define PADDING_SIZE_Y 0
#define PADDING_SIZE_Z 0
#define DILATION_SIZE_X 1
#define DILATION_SIZE_Y 1
#define DILATION_SIZE_Z 1
#define FILTER_ARRAY_NUM 1
#define INPUT0_OFFSET_WITH_PADDING 0
#define GROUPED 0
#define SKIP_BATCH 1
#define FUSED_OP0_LOAD_VEC \
	MAKE_VECTOR_TYPE(FUSED_OP_0_INPUT0_TYPE,8) eltwise0_data0 = as_half8( _sub_group_block_read_us8((const __global ushort*)(eltwise0_input0 + FUSED_OP_0_INPUT0_GET_INDEX_SAFE(0,(feature_block * 16),y,x))));
#define FUSED_OP0_ACTION_VEC \
	half8 dst_out_0_tmp = dst+eltwise0_data0;\
	half8 dst_out_0 = convert_half8(dst_out_0_tmp);
#define FUSED_OP1_LOAD_VEC 
#define ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE half
#define ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_MIN -ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_MAX
#define ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_ONE 1.0h
#define ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_FUSED_OP1_VEC_MAX_FUNC fmax
#define ACTIVATION_FUNC_FUSED_OP1_VEC_MIN_FUNC fmin
#define ACTIVATION_FUNC_FUSED_OP1_VEC_ABS_FUNC fabs
#define ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE_SIZE 2
#define ACTIVATION_FUNC_FUSED_OP1_VEC_IS_FP 1
#define ACTIVATION_PARAMS_FUSED_OP1_VEC NL_M_FUSED_OP1_VEC, NL_N_FUSED_OP1_VEC
#define ACTIVATION_FUNC_FUSED_OP1_VEC(input, m, n) (ACTIVATION_FUNC_FUSED_OP1_VEC_MAX_FUNC(ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_ZERO, input))
#define ACTIVATION_FUSED_OP1_VEC(input, params) ACTIVATION_FUNC_FUSED_OP1_VEC(input, params)
#define FUSED_OP1_ACTION_VEC \
	half8 dst_out_1 = convert_half8(dst_out_0);\
	dst_out_1 = ACTIVATION_FUNC_FUSED_OP1_VEC(dst_out_1, (half8)(as_float(0x0)/*0.000000e+00*/),(half8)(as_float(0x0)/*0.000000e+00*/));
#define FUSED_OPS_VEC \
	FUSED_OP0_LOAD_VEC\
	FUSED_OP0_ACTION_VEC\
	FUSED_OP1_LOAD_VEC\
	FUSED_OP1_ACTION_VEC
#define FUSED_OPS_PRELOAD_VEC \
	FUSED_OP0_LOAD_VEC\
	FUSED_OP1_LOAD_VEC
#define FUSED_OPS_CALC_VEC \
	FUSED_OP0_ACTION_VEC\
	FUSED_OP1_ACTION_VEC
#define FUSED_OPS_RESULT_VEC dst_out_1
#define FUSED_OPS_CAN_USE_PRELOAD_VEC 1
#define FUSED_OP0_LOAD_SCALAR \
	FUSED_OP_0_INPUT0_TYPE eltwise0_data0 = (half)(as_half( _sub_group_block_read_us((const __global ushort*)(eltwise0_input0 + FUSED_OP_0_INPUT0_GET_INDEX_SAFE(0,(feature_block * 16),yi,xi)))));
#define FUSED_OP0_ACTION_SCALAR \
	half dst_i__out_0_tmp = dst[i]+eltwise0_data0;\
	half dst_i__out_0 = convert_half(dst_i__out_0_tmp);
#define FUSED_OP1_LOAD_SCALAR 
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE half
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_MIN -ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_MAX
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_ONE 1.0h
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_MAX_FUNC fmax
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_MIN_FUNC fmin
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_ABS_FUNC fabs
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE_SIZE 2
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_IS_FP 1
#define ACTIVATION_PARAMS_FUSED_OP1_SCALAR NL_M_FUSED_OP1_SCALAR, NL_N_FUSED_OP1_SCALAR
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR(input, m, n) (ACTIVATION_FUNC_FUSED_OP1_SCALAR_MAX_FUNC(ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_ZERO, input))
#define ACTIVATION_FUSED_OP1_SCALAR(input, params) ACTIVATION_FUNC_FUSED_OP1_SCALAR(input, params)
#define FUSED_OP1_ACTION_SCALAR \
	half dst_i__out_1 = convert_half(dst_i__out_0);\
	dst_i__out_1 = ACTIVATION_FUNC_FUSED_OP1_SCALAR(dst_i__out_1, (half)(as_float(0x0)/*0.000000e+00*/),(half)(as_float(0x0)/*0.000000e+00*/));
#define FUSED_OPS_SCALAR \
	FUSED_OP0_LOAD_SCALAR\
	FUSED_OP0_ACTION_SCALAR\
	FUSED_OP1_LOAD_SCALAR\
	FUSED_OP1_ACTION_SCALAR
#define FUSED_OPS_PRELOAD_SCALAR \
	FUSED_OP0_LOAD_SCALAR\
	FUSED_OP1_LOAD_SCALAR
#define FUSED_OPS_CALC_SCALAR \
	FUSED_OP0_ACTION_SCALAR\
	FUSED_OP1_ACTION_SCALAR
#define FUSED_OPS_RESULT_SCALAR dst_i__out_1
#define FUSED_OPS_CAN_USE_PRELOAD_SCALAR 1
#define FUSED_OP0_LOAD_SCALAR_B1 \
	FUSED_OP_0_INPUT0_TYPE eltwise0_data0 = (half)(as_half( _sub_group_block_read_us((const __global ushort*)(eltwise0_input0 + FUSED_OP_0_INPUT0_GET_INDEX_SAFE(0,(feature_block * 16),0,0)))));
#define FUSED_OP0_ACTION_SCALAR_B1 \
	half dst_out_0_tmp = dst+eltwise0_data0;\
	half dst_out_0 = convert_half(dst_out_0_tmp);
#define FUSED_OP1_LOAD_SCALAR_B1 
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE half
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_MAX HALF_MAX
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_MIN -ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_MAX
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_ONE 1.0h
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_ZERO 0.0h
#define TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE(v) convert_half(v)
#define TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE_SAT(v) convert_half(v)
#define AS_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE(v) as_half(v)
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_MAX_FUNC fmax
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_MIN_FUNC fmin
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_ABS_FUNC fabs
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE_SIZE 2
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_IS_FP 1
#define ACTIVATION_PARAMS_FUSED_OP1_SCALAR_B1 NL_M_FUSED_OP1_SCALAR_B1, NL_N_FUSED_OP1_SCALAR_B1
#define ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1(input, m, n) (ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_MAX_FUNC(ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_ZERO, input))
#define ACTIVATION_FUSED_OP1_SCALAR_B1(input, params) ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1(input, params)
#define FUSED_OP1_ACTION_SCALAR_B1 \
	half dst_out_1 = convert_half(dst_out_0);\
	dst_out_1 = ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1(dst_out_1, (half)(as_float(0x0)/*0.000000e+00*/),(half)(as_float(0x0)/*0.000000e+00*/));
#define FUSED_OPS_SCALAR_B1 \
	FUSED_OP0_LOAD_SCALAR_B1\
	FUSED_OP0_ACTION_SCALAR_B1\
	FUSED_OP1_LOAD_SCALAR_B1\
	FUSED_OP1_ACTION_SCALAR_B1
#define FUSED_OPS_PRELOAD_SCALAR_B1 \
	FUSED_OP0_LOAD_SCALAR_B1\
	FUSED_OP1_LOAD_SCALAR_B1
#define FUSED_OPS_CALC_SCALAR_B1 \
	FUSED_OP0_ACTION_SCALAR_B1\
	FUSED_OP1_ACTION_SCALAR_B1
#define FUSED_OPS_RESULT_SCALAR_B1 dst_out_1
#define FUSED_OPS_CAN_USE_PRELOAD_SCALAR_B1 1
#define FUSED_OP_0_INPUT0_SIZE_X 14
#define FUSED_OP_0_INPUT0_SIZE_Y 14
#define FUSED_OP_0_INPUT0_SIZE_Z 1
#define FUSED_OP_0_INPUT0_SIZE_W 1
#define FUSED_OP_0_INPUT0_SIZE_U 1
#define FUSED_OP_0_INPUT0_SIZE_V 1
#define FUSED_OP_0_INPUT0_FEATURE_NUM 1024
#define FUSED_OP_0_INPUT0_BATCH_NUM 1
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_X 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Y 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Z 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_W 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_U 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_V 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_FEATURE_NUM 0
#define FUSED_OP_0_INPUT0_PAD_BEFORE_BATCH_NUM 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_X 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Y 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Z 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_W 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_U 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_V 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_FEATURE_NUM 0
#define FUSED_OP_0_INPUT0_PAD_AFTER_BATCH_NUM 0
#define FUSED_OP_0_INPUT0_LENGTH 200704
#define FUSED_OP_0_INPUT0_X_PITCH 1
#define FUSED_OP_0_INPUT0_Y_PITCH 14
#define FUSED_OP_0_INPUT0_Z_PITCH 1
#define FUSED_OP_0_INPUT0_W_PITCH 1
#define FUSED_OP_0_INPUT0_U_PITCH 1
#define FUSED_OP_0_INPUT0_V_PITCH 1
#define FUSED_OP_0_INPUT0_FEATURE_PITCH 196
#define FUSED_OP_0_INPUT0_BATCH_PITCH 200704
#define FUSED_OP_0_INPUT0_GET_INDEX_SAFE(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX_SAFE(FUSED_OP_0_INPUT0, b, f, y, x)
#define FUSED_OP_0_INPUT0_GET_INDEX(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(FUSED_OP_0_INPUT0, b, f, y, x)
#define FUSED_OP_0_INPUT0_GET_INDEX_RAW(b, f, y, x) GET_DATA_B_FS_YX_FSV16_INDEX(FUSED_OP_0_INPUT0, b, f, y, x)
#define FUSED_OP_0_INPUT0_VIEW_OFFSET 0
#define FUSED_OP_0_INPUT0_DIMS 4
#define FUSED_OP_0_INPUT0_SIMPLE 0
#define FUSED_OP_0_INPUT0_GROUPED 0
#define FUSED_OP_0_INPUT0_LAYOUT_B_FS_YX_FSV16 1
#define FUSED_OP_0_INPUT0_TYPE half
#define FUSED_OP_0_INPUT0_VAL_MAX HALF_MAX
#define FUSED_OP_0_INPUT0_VAL_MIN -FUSED_OP_0_INPUT0_VAL_MAX
#define FUSED_OP_0_INPUT0_VAL_ONE 1.0h
#define FUSED_OP_0_INPUT0_VAL_ZERO 0.0h
#define TO_FUSED_OP_0_INPUT0_TYPE(v) convert_half(v)
#define TO_FUSED_OP_0_INPUT0_TYPE_SAT(v) convert_half(v)
#define AS_FUSED_OP_0_INPUT0_TYPE(v) as_half(v)
#define FUSED_OP_0_INPUT0_MAX_FUNC fmax
#define FUSED_OP_0_INPUT0_MIN_FUNC fmin
#define FUSED_OP_0_INPUT0_ABS_FUNC fabs
#define FUSED_OP_0_INPUT0_TYPE_SIZE 2
#define FUSED_OP_0_INPUT0_IS_FP 1
#define FUSED_OP_0_INPUT0_OFFSET 0
#define FUSED_OP_0_INPUT0_SIZES_DATA { 14,14,1024,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(FUSED_OP_0_INPUT0_SIZES) = FUSED_OP_0_INPUT0_SIZES_DATA;
#define FUSED_OP_0_INPUT0_SIZES CONST_ARRAY_REF(FUSED_OP_0_INPUT0_SIZES)
#define FUSED_OP_0_INPUT0_PITCHES (size_t []){ 1,14,196,200704,1,1,1,1,1, } 
#define FUSED_OP_0_INPUT0_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP_0_INPUT0_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP_0_OUTPUT_SIZE_X 14
#define FUSED_OP_0_OUTPUT_SIZE_Y 14
#define FUSED_OP_0_OUTPUT_SIZE_Z 1
#define FUSED_OP_0_OUTPUT_SIZE_W 1
#define FUSED_OP_0_OUTPUT_SIZE_U 1
#define FUSED_OP_0_OUTPUT_SIZE_V 1
#define FUSED_OP_0_OUTPUT_FEATURE_NUM 1024
#define FUSED_OP_0_OUTPUT_BATCH_NUM 1
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_X 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Y 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Z 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_W 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_U 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_V 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_FEATURE_NUM 0
#define FUSED_OP_0_OUTPUT_PAD_BEFORE_BATCH_NUM 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_X 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Y 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Z 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_W 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_U 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_V 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_FEATURE_NUM 0
#define FUSED_OP_0_OUTPUT_PAD_AFTER_BATCH_NUM 0
#define FUSED_OP_0_OUTPUT_LENGTH 200704
#define FUSED_OP_0_OUTPUT_X_PITCH 1
#define FUSED_OP_0_OUTPUT_Y_PITCH 14
#define FUSED_OP_0_OUTPUT_Z_PITCH 1
#define FUSED_OP_0_OUTPUT_W_PITCH 1
#define FUSED_OP_0_OUTPUT_U_PITCH 1
#define FUSED_OP_0_OUTPUT_V_PITCH 1
#define FUSED_OP_0_OUTPUT_FEATURE_PITCH 196
#define FUSED_OP_0_OUTPUT_BATCH_PITCH 200704
#define FUSED_OP_0_OUTPUT_GET_INDEX_SAFE(b, f, y, x) GET_DATA_INDEX_SAFE(FUSED_OP_0_OUTPUT, b, f, y, x)
#define FUSED_OP_0_OUTPUT_GET_INDEX(b, f, y, x) GET_DATA_INDEX(FUSED_OP_0_OUTPUT, b, f, y, x)
#define FUSED_OP_0_OUTPUT_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(FUSED_OP_0_OUTPUT, b, f, y, x)
#define FUSED_OP_0_OUTPUT_VIEW_OFFSET 0
#define FUSED_OP_0_OUTPUT_DIMS 4
#define FUSED_OP_0_OUTPUT_SIMPLE 1
#define FUSED_OP_0_OUTPUT_GROUPED 0
#define FUSED_OP_0_OUTPUT_LAYOUT_BFYX 1
#define FUSED_OP_0_OUTPUT_TYPE half
#define FUSED_OP_0_OUTPUT_VAL_MAX HALF_MAX
#define FUSED_OP_0_OUTPUT_VAL_MIN -FUSED_OP_0_OUTPUT_VAL_MAX
#define FUSED_OP_0_OUTPUT_VAL_ONE 1.0h
#define FUSED_OP_0_OUTPUT_VAL_ZERO 0.0h
#define TO_FUSED_OP_0_OUTPUT_TYPE(v) convert_half(v)
#define TO_FUSED_OP_0_OUTPUT_TYPE_SAT(v) convert_half(v)
#define AS_FUSED_OP_0_OUTPUT_TYPE(v) as_half(v)
#define FUSED_OP_0_OUTPUT_MAX_FUNC fmax
#define FUSED_OP_0_OUTPUT_MIN_FUNC fmin
#define FUSED_OP_0_OUTPUT_ABS_FUNC fabs
#define FUSED_OP_0_OUTPUT_TYPE_SIZE 2
#define FUSED_OP_0_OUTPUT_IS_FP 1
#define FUSED_OP_0_OUTPUT_OFFSET 0
#define FUSED_OP_0_OUTPUT_SIZES_DATA { 14,14,1024,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(FUSED_OP_0_OUTPUT_SIZES) = FUSED_OP_0_OUTPUT_SIZES_DATA;
#define FUSED_OP_0_OUTPUT_SIZES CONST_ARRAY_REF(FUSED_OP_0_OUTPUT_SIZES)
#define FUSED_OP_0_OUTPUT_PITCHES (size_t []){ 1,14,196,200704,1,1,1,1,1, } 
#define FUSED_OP_0_OUTPUT_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP_0_OUTPUT_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP0_DECLS \
	const __global half* eltwise0_input0
#define FUSED_OP0_ARGS \
	eltwise0_input0
#define FUSED_OP_1_OUTPUT_SIZE_X 14
#define FUSED_OP_1_OUTPUT_SIZE_Y 14
#define FUSED_OP_1_OUTPUT_SIZE_Z 1
#define FUSED_OP_1_OUTPUT_SIZE_W 1
#define FUSED_OP_1_OUTPUT_SIZE_U 1
#define FUSED_OP_1_OUTPUT_SIZE_V 1
#define FUSED_OP_1_OUTPUT_FEATURE_NUM 1024
#define FUSED_OP_1_OUTPUT_BATCH_NUM 1
#define FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_X 0
#define FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_Y 0
#define FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_Z 0
#define FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_W 0
#define FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_U 0
#define FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_V 0
#define FUSED_OP_1_OUTPUT_PAD_BEFORE_FEATURE_NUM 0
#define FUSED_OP_1_OUTPUT_PAD_BEFORE_BATCH_NUM 0
#define FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_X 0
#define FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_Y 0
#define FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_Z 0
#define FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_W 0
#define FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_U 0
#define FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_V 0
#define FUSED_OP_1_OUTPUT_PAD_AFTER_FEATURE_NUM 0
#define FUSED_OP_1_OUTPUT_PAD_AFTER_BATCH_NUM 0
#define FUSED_OP_1_OUTPUT_LENGTH 200704
#define FUSED_OP_1_OUTPUT_X_PITCH 1
#define FUSED_OP_1_OUTPUT_Y_PITCH 14
#define FUSED_OP_1_OUTPUT_Z_PITCH 1
#define FUSED_OP_1_OUTPUT_W_PITCH 1
#define FUSED_OP_1_OUTPUT_U_PITCH 1
#define FUSED_OP_1_OUTPUT_V_PITCH 1
#define FUSED_OP_1_OUTPUT_FEATURE_PITCH 196
#define FUSED_OP_1_OUTPUT_BATCH_PITCH 200704
#define FUSED_OP_1_OUTPUT_GET_INDEX_SAFE(b, f, y, x) GET_DATA_INDEX_SAFE(FUSED_OP_1_OUTPUT, b, f, y, x)
#define FUSED_OP_1_OUTPUT_GET_INDEX(b, f, y, x) GET_DATA_INDEX(FUSED_OP_1_OUTPUT, b, f, y, x)
#define FUSED_OP_1_OUTPUT_GET_INDEX_RAW(b, f, y, x) GET_DATA_INDEX_RAW(FUSED_OP_1_OUTPUT, b, f, y, x)
#define FUSED_OP_1_OUTPUT_VIEW_OFFSET 0
#define FUSED_OP_1_OUTPUT_DIMS 4
#define FUSED_OP_1_OUTPUT_SIMPLE 1
#define FUSED_OP_1_OUTPUT_GROUPED 0
#define FUSED_OP_1_OUTPUT_LAYOUT_BFYX 1
#define FUSED_OP_1_OUTPUT_TYPE half
#define FUSED_OP_1_OUTPUT_VAL_MAX HALF_MAX
#define FUSED_OP_1_OUTPUT_VAL_MIN -FUSED_OP_1_OUTPUT_VAL_MAX
#define FUSED_OP_1_OUTPUT_VAL_ONE 1.0h
#define FUSED_OP_1_OUTPUT_VAL_ZERO 0.0h
#define TO_FUSED_OP_1_OUTPUT_TYPE(v) convert_half(v)
#define TO_FUSED_OP_1_OUTPUT_TYPE_SAT(v) convert_half(v)
#define AS_FUSED_OP_1_OUTPUT_TYPE(v) as_half(v)
#define FUSED_OP_1_OUTPUT_MAX_FUNC fmax
#define FUSED_OP_1_OUTPUT_MIN_FUNC fmin
#define FUSED_OP_1_OUTPUT_ABS_FUNC fabs
#define FUSED_OP_1_OUTPUT_TYPE_SIZE 2
#define FUSED_OP_1_OUTPUT_IS_FP 1
#define FUSED_OP_1_OUTPUT_OFFSET 0
#define FUSED_OP_1_OUTPUT_SIZES_DATA { 14,14,1024,1,1,1,1,1,1, } 
CONST_ARRAY_DECL(FUSED_OP_1_OUTPUT_SIZES) = FUSED_OP_1_OUTPUT_SIZES_DATA;
#define FUSED_OP_1_OUTPUT_SIZES CONST_ARRAY_REF(FUSED_OP_1_OUTPUT_SIZES)
#define FUSED_OP_1_OUTPUT_PITCHES (size_t []){ 1,14,196,200704,1,1,1,1,1, } 
#define FUSED_OP_1_OUTPUT_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP_1_OUTPUT_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, } 
#define FUSED_OP1_DECLS 
#define FUSED_OP1_ARGS 
#define FUSED_OPS_DECLS \
	FUSED_OP0_DECLS
#define FUSED_OPS_ARGS \
	FUSED_OP0_ARGS
#define HAS_FUSED_OPS 1
#define HAS_FUSED_OPS_DECLS 1
#define X_BLOCK_SIZE 8
#define SLM_DIV_FACTOR 2
#define SUB_GROUP_SIZE 16
#define WORK_GROUP_SIZE 32
#define PADDED_INPUT 0
#define PADDED_OUTPUT 0
#define NON_UNIT_FUSED_OP_SPATIAL 1
#define IC_BLOCKS 16
#define INPUT_LEFTOVERS 0


#ifndef UNIT_TYPE
#error Unit type not defined.
#endif
#if UNIT_TYPE_SIZE == 2
#define UNIT_BLOCK_RW_TYPE ushort
#define UNIT_BLOCK_READ_FUNC _sub_group_block_read_us
#define UNIT_BLOCK_WRITE_FUNC _sub_group_block_write_us
#elif UNIT_TYPE_SIZE == 4
#define UNIT_BLOCK_RW_TYPE uint
#define UNIT_BLOCK_READ_FUNC _sub_group_block_read
#define UNIT_BLOCK_WRITE_FUNC _sub_group_block_write
#else
#error Unsupported unit type for block read/write.
#endif
#define UNIT_TYPE2 MAKE_VECTOR_TYPE(UNIT_TYPE, 2)
#define UNIT_TYPE4 MAKE_VECTOR_TYPE(UNIT_TYPE, 4)
#define UNIT_TYPE8 MAKE_VECTOR_TYPE(UNIT_TYPE, 8)
#define UNIT_BLOCK_RW_TYPE2 MAKE_VECTOR_TYPE(UNIT_BLOCK_RW_TYPE, 2)
#define UNIT_BLOCK_RW_TYPE4 MAKE_VECTOR_TYPE(UNIT_BLOCK_RW_TYPE, 4)
#define UNIT_BLOCK_RW_TYPE8 MAKE_VECTOR_TYPE(UNIT_BLOCK_RW_TYPE, 8)
#define UNIT_BLOCK_READ_FUNC2 CAT(UNIT_BLOCK_READ_FUNC, 2)
#define UNIT_BLOCK_READ_FUNC4 CAT(UNIT_BLOCK_READ_FUNC, 4)
#define UNIT_BLOCK_READ_FUNC8 CAT(UNIT_BLOCK_READ_FUNC, 8)
#define UNIT_BLOCK_WRITE_FUNC2 CAT(UNIT_BLOCK_WRITE_FUNC, 2)
#define UNIT_BLOCK_WRITE_FUNC4 CAT(UNIT_BLOCK_WRITE_FUNC, 4)
#define UNIT_BLOCK_WRITE_FUNC8 CAT(UNIT_BLOCK_WRITE_FUNC, 8)
#define UNIT_BLOCK_READ(ptr, offset) AS_TYPE(UNIT_TYPE, UNIT_BLOCK_READ_FUNC( (const __global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset)))
#define UNIT_BLOCK_READ2(ptr, offset) AS_TYPE(UNIT_TYPE2, UNIT_BLOCK_READ_FUNC2((const __global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset)))
#define UNIT_BLOCK_READ4(ptr, offset) AS_TYPE(UNIT_TYPE4, UNIT_BLOCK_READ_FUNC4((const __global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset)))
#define UNIT_BLOCK_READ8(ptr, offset) AS_TYPE(UNIT_TYPE8, UNIT_BLOCK_READ_FUNC8((const __global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset)))
#define UNIT_BLOCK_WRITE(ptr, offset, val) UNIT_BLOCK_WRITE_FUNC( (__global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset), AS_TYPE(UNIT_BLOCK_RW_TYPE, val))
#define UNIT_BLOCK_WRITE2(ptr, offset, val) UNIT_BLOCK_WRITE_FUNC2((__global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset), AS_TYPE(UNIT_BLOCK_RW_TYPE2, val))
#define UNIT_BLOCK_WRITE4(ptr, offset, val) UNIT_BLOCK_WRITE_FUNC4((__global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset), AS_TYPE(UNIT_BLOCK_RW_TYPE4, val))
#define UNIT_BLOCK_WRITE8(ptr, offset, val) UNIT_BLOCK_WRITE_FUNC8((__global UNIT_BLOCK_RW_TYPE*)(ptr) + (offset), AS_TYPE(UNIT_BLOCK_RW_TYPE8, val))
#if X_BLOCK_SIZE > 1
# define GET_SRC(data, id) AS_TYPE(MAKE_VECTOR_TYPE(UNIT_TYPE, X_BLOCK_SIZE), \
 _sub_group_shuffle( \
 AS_TYPE(MAKE_VECTOR_TYPE(UNIT_BLOCK_RW_TYPE, X_BLOCK_SIZE), data), \
 id))
#else
# define GET_SRC(data, id) AS_TYPE(UNIT_TYPE, _sub_group_shuffle(AS_TYPE(UNIT_BLOCK_RW_TYPE, data), id))
#endif
#define FEATURE_SLICE_SIZE 16
#if X_BLOCK_SIZE > 1
# define UNIT_BLOCK_READ_VEC(ptr, offset) CAT(UNIT_BLOCK_READ, X_BLOCK_SIZE)(ptr, offset)
# define UNIT_BLOCK_WRITE_VEC(ptr, offset, val) CAT(UNIT_BLOCK_WRITE, X_BLOCK_SIZE)(ptr, offset, val)
#endif
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, SUB_GROUP_SIZE * SLM_DIV_FACTOR, 1)))
KERNEL(convolution_b_fs_yx_fsv16_1x1)(
 OPTIONAL_SHAPE_INFO_ARG
 __global INPUT0_TYPE* input,
 __global OUTPUT_TYPE* output,
 __global FILTER_TYPE* weights
#if BIAS_TERM
 , __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
 , FUSED_OPS_DECLS
#endif
) {
#if X_BLOCK_SIZE > 1
 const uint xy = (int)get_global_id(0);
 const uint x = (xy * X_BLOCK_SIZE) % OUTPUT_SIZE_X;
 const uint y = (xy * X_BLOCK_SIZE) / OUTPUT_SIZE_X;
 const uint input_x = x;
 const uint input_y = y;
#endif
 const uint b = (int)get_global_id(2);
 const uint sglid = (int)get_sub_group_local_id();
 const uint lid1 = (int)get_local_id(1);
 const uint feature_per_wg = (int)get_local_size(1) / SLM_DIV_FACTOR;
 const uint feature_sub_block = lid1 / feature_per_wg;
 const uint feature_block = (int)get_group_id(1);
 const uint input_x_pitch = FEATURE_SLICE_SIZE;
 const uint input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
 const uint input_fs_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
 const uint input_total_f_size = INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
 const uint input_b_pitch = input_fs_pitch * ((input_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
 const uint input_fs_pad_before = INPUT0_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;
 const uint input_offset = b * input_b_pitch +
 input_fs_pad_before * input_fs_pitch +
 INPUT0_PAD_BEFORE_SIZE_Y * input_y_pitch +
 INPUT0_PAD_BEFORE_SIZE_X * input_x_pitch;
 const uint output_x_pitch = FEATURE_SLICE_SIZE;
 const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
 const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
 const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
 const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
 const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;
 const uint output_offset = b * output_b_pitch +
 (feature_block + output_fs_pad_before) * output_fs_pitch +
 (OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
 (OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;
 const uint filter_isv_pitch = FEATURE_SLICE_SIZE;
 const uint filter_x_pitch = FEATURE_SLICE_SIZE * FEATURE_SLICE_SIZE;
 const uint filter_y_pitch = filter_x_pitch * FILTER_SIZE_X;
 const uint filter_is_pitch = filter_y_pitch * FILTER_SIZE_Y;
 const uint filter_os_pitch = filter_is_pitch * ((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
 const uint filter_offset = feature_block * filter_os_pitch;
#if X_BLOCK_SIZE > 1
 typedef MAKE_VECTOR_TYPE(UNIT_TYPE, X_BLOCK_SIZE) vec_t;
#else
 typedef UNIT_TYPE vec_t;
#endif
#if BIAS_TERM
#if SLM_DIV_FACTOR == 1
 vec_t dst = (vec_t)(UNIT_BLOCK_READ(biases, feature_block * FEATURE_SLICE_SIZE));
#else
 vec_t dst;
 if (feature_sub_block == 0) {
 dst = (vec_t)(UNIT_BLOCK_READ(biases, feature_block * FEATURE_SLICE_SIZE));
 } else {
 dst = UNIT_VAL_ZERO;
 }
#endif
#else
 vec_t dst = UNIT_VAL_ZERO;
#endif
#if SLM_DIV_FACTOR > 1
 __local vec_t partial_summ[WORK_GROUP_SIZE];
 for (uint k = feature_sub_block * IC_BLOCKS / SLM_DIV_FACTOR; k < (feature_sub_block + 1) * IC_BLOCKS / SLM_DIV_FACTOR; k++)
 {
#else
 for (uint k = 0; k < IC_BLOCKS; k++)
 {
#endif
 vec_t src = 0;
 if (INPUT_LEFTOVERS && ((k + 1) * FEATURE_SLICE_SIZE >= INPUT0_FEATURE_NUM))
 {
 if (k * FEATURE_SLICE_SIZE + sglid < INPUT0_FEATURE_NUM)
 {
#if X_BLOCK_SIZE > 1
 __attribute__((opencl_unroll_hint(X_BLOCK_SIZE)))
 for (int i = 0; i < X_BLOCK_SIZE; i++)
 {
 const uint xb = (x + i) % INPUT0_SIZE_X;
 const uint yb = y + (x + i) / INPUT0_SIZE_X;
 const uint input_idx = input_offset + k * input_fs_pitch + yb * input_y_pitch + xb * input_x_pitch;
 src[i] = input[input_idx + sglid];
 }
#else
 src = input[input_offset + k * input_fs_pitch + sglid];
#endif
 }
 }
 else
 {
#if PADDED_INPUT
#if X_BLOCK_SIZE > 1
 __attribute__((opencl_unroll_hint(X_BLOCK_SIZE)))
 for (int i = 0; i < X_BLOCK_SIZE; i++)
 {
 const uint xb = (x + i) % INPUT0_SIZE_X;
 const uint yb = y + (x + i) / INPUT0_SIZE_X;
 const uint input_idx = input_offset + k * input_fs_pitch + yb * input_y_pitch + xb * input_x_pitch;
 src[i] = UNIT_BLOCK_READ(input, input_idx);
 }
#else
 src = UNIT_BLOCK_READ(input, input_offset + k * input_fs_pitch);
#endif
#else
#if X_BLOCK_SIZE > 1
 src = UNIT_BLOCK_READ_VEC(input, input_offset + k * input_fs_pitch + input_y * input_y_pitch + input_x * input_x_pitch);
#else
 src = UNIT_BLOCK_READ(input, input_offset + k * input_fs_pitch);
#endif
#endif
 }
 UNIT_TYPE8 wei0 = UNIT_BLOCK_READ8(weights, filter_offset + k * filter_is_pitch);
 UNIT_TYPE8 wei1 = UNIT_BLOCK_READ8(weights, filter_offset + k * filter_is_pitch + 8 * filter_isv_pitch);
 const vec_t src0 = GET_SRC(src, 0);
 const vec_t src1 = GET_SRC(src, 1);
 const vec_t src2 = GET_SRC(src, 2);
 const vec_t src3 = GET_SRC(src, 3);
 const vec_t src4 = GET_SRC(src, 4);
 const vec_t src5 = GET_SRC(src, 5);
 const vec_t src6 = GET_SRC(src, 6);
 const vec_t src7 = GET_SRC(src, 7);
 const vec_t src8 = GET_SRC(src, 8);
 const vec_t src9 = GET_SRC(src, 9);
 const vec_t src10 = GET_SRC(src, 10);
 const vec_t src11 = GET_SRC(src, 11);
 const vec_t src12 = GET_SRC(src, 12);
 const vec_t src13 = GET_SRC(src, 13);
 const vec_t src14 = GET_SRC(src, 14);
 const vec_t src15 = GET_SRC(src, 15);
 dst = mad(wei0.s0, src0, dst);
 dst = mad(wei0.s1, src1, dst);
 dst = mad(wei0.s2, src2, dst);
 dst = mad(wei0.s3, src3, dst);
 dst = mad(wei0.s4, src4, dst);
 dst = mad(wei0.s5, src5, dst);
 dst = mad(wei0.s6, src6, dst);
 dst = mad(wei0.s7, src7, dst);
 dst = mad(wei1.s0, src8, dst);
 dst = mad(wei1.s1, src9, dst);
 dst = mad(wei1.s2, src10, dst);
 dst = mad(wei1.s3, src11, dst);
 dst = mad(wei1.s4, src12, dst);
 dst = mad(wei1.s5, src13, dst);
 dst = mad(wei1.s6, src14, dst);
 dst = mad(wei1.s7, src15, dst);
 }
#if SLM_DIV_FACTOR > 1
 partial_summ[lid1] = dst;
 barrier(CLK_LOCAL_MEM_FENCE);
 if (feature_sub_block == 0) {
 unroll_for(int i = 1; i < SLM_DIV_FACTOR; i++)
 dst += partial_summ[lid1 % feature_per_wg + i * feature_per_wg];
#endif
 dst = ACTIVATION(dst, ACTIVATION_PARAMS);
#if OUTPUT_LEFTOVERS
 if ((feature_block + 1) * FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM)
 {
#if X_BLOCK_SIZE > 1
 for (int i = 0; i < X_BLOCK_SIZE; i++) {
 if (xy * X_BLOCK_SIZE + i >= OUTPUT_SIZE_X * OUTPUT_SIZE_Y || feature_block * FEATURE_SLICE_SIZE + sglid >= OUTPUT_FEATURE_NUM)
 return;
 int xi = (x + i) % OUTPUT_SIZE_X;
 int yi = y + ((x + i) / OUTPUT_SIZE_X);
#if HAS_FUSED_OPS
 FUSED_OPS_SCALAR;
 dst[i] = FUSED_OPS_RESULT_SCALAR;
#endif
 output[output_offset + yi * output_y_pitch + xi * output_x_pitch + sglid] = dst[i];
 }
#else
 if (feature_block * FEATURE_SLICE_SIZE + sglid >= OUTPUT_FEATURE_NUM)
 return;
#if HAS_FUSED_OPS
 FUSED_OPS_SCALAR_B1;
 dst = FUSED_OPS_RESULT_SCALAR_B1;
#endif
 output[output_offset + sglid] = dst;
#endif
 }
 else
#endif
#if X_BLOCK_SIZE > 1
 {
#if !PADDED_OUTPUT && !NON_UNIT_FUSED_OP_SPATIAL
 if (xy * X_BLOCK_SIZE + X_BLOCK_SIZE <= OUTPUT_SIZE_X * OUTPUT_SIZE_Y || (OUTPUT_SIZE_X * OUTPUT_SIZE_Y) % X_BLOCK_SIZE == 0) {
#else
 if (x + X_BLOCK_SIZE <= OUTPUT_SIZE_X || OUTPUT_SIZE_X % X_BLOCK_SIZE == 0) {
#endif
#if HAS_FUSED_OPS
 FUSED_OPS_VEC;
 dst = FUSED_OPS_RESULT_VEC;
#endif
 UNIT_BLOCK_WRITE_VEC(output, output_offset + y * output_y_pitch + x * output_x_pitch, dst);
 } else {
 for (int i = 0; i < X_BLOCK_SIZE; i++) {
 if (xy * X_BLOCK_SIZE + i >= OUTPUT_SIZE_X * OUTPUT_SIZE_Y)
 return;
 int xi = (x + i) % OUTPUT_SIZE_X;
 int yi = y + ((x + i) / OUTPUT_SIZE_X);
#if HAS_FUSED_OPS
 FUSED_OPS_SCALAR;
 dst[i] = FUSED_OPS_RESULT_SCALAR;
#endif
 UNIT_BLOCK_WRITE(output, output_offset + yi * output_y_pitch + xi * output_x_pitch, dst[i]);
 }
 }
 }
#else
 {
#if HAS_FUSED_OPS
 FUSED_OPS_SCALAR_B1;
 dst = FUSED_OPS_RESULT_SCALAR_B1;
#endif
 UNIT_BLOCK_WRITE(output, output_offset, dst);
 }
#endif
#if SLM_DIV_FACTOR > 1
 }
#endif
}
#undef GET_SRC
#undef FEATURE_SLICE_SIZE
#undef UNIT_BLOCK_READ_VEC
#undef UNIT_BLOCK_WRITE_VEC
#ifdef FEATURE_SLICE_SIZE
#undef FEATURE_SLICE_SIZE
#endif
#ifdef UNIT_BLOCK_RW_TYPE
#undef UNIT_BLOCK_RW_TYPE
#endif
#ifdef UNIT_BLOCK_READ_FUNC
#undef UNIT_BLOCK_READ_FUNC
#endif
#ifdef UNIT_BLOCK_WRITE_FUNC
#undef UNIT_BLOCK_WRITE_FUNC
#endif
#ifdef UNIT_BLOCK_RW_TYPE
#undef UNIT_BLOCK_RW_TYPE
#endif
#ifdef UNIT_BLOCK_READ_FUNC
#undef UNIT_BLOCK_READ_FUNC
#endif
#ifdef UNIT_BLOCK_WRITE_FUNC
#undef UNIT_BLOCK_WRITE_FUNC
#endif
#ifdef UNIT_TYPE2
#undef UNIT_TYPE2
#endif
#ifdef UNIT_TYPE4
#undef UNIT_TYPE4
#endif
#ifdef UNIT_TYPE8
#undef UNIT_TYPE8
#endif
#ifdef UNIT_BLOCK_RW_TYPE2
#undef UNIT_BLOCK_RW_TYPE2
#endif
#ifdef UNIT_BLOCK_RW_TYPE4
#undef UNIT_BLOCK_RW_TYPE4
#endif
#ifdef UNIT_BLOCK_RW_TYPE8
#undef UNIT_BLOCK_RW_TYPE8
#endif
#ifdef UNIT_BLOCK_READ_FUNC2
#undef UNIT_BLOCK_READ_FUNC2
#endif
#ifdef UNIT_BLOCK_READ_FUNC4
#undef UNIT_BLOCK_READ_FUNC4
#endif
#ifdef UNIT_BLOCK_READ_FUNC8
#undef UNIT_BLOCK_READ_FUNC8
#endif
#ifdef UNIT_BLOCK_WRITE_FUNC2
#undef UNIT_BLOCK_WRITE_FUNC2
#endif
#ifdef UNIT_BLOCK_WRITE_FUNC4
#undef UNIT_BLOCK_WRITE_FUNC4
#endif
#ifdef UNIT_BLOCK_WRITE_FUNC8
#undef UNIT_BLOCK_WRITE_FUNC8
#endif
#ifdef UNIT_BLOCK_READ
#undef UNIT_BLOCK_READ
#endif
#ifdef UNIT_BLOCK_READ2
#undef UNIT_BLOCK_READ2
#endif
#ifdef UNIT_BLOCK_READ4
#undef UNIT_BLOCK_READ4
#endif
#ifdef UNIT_BLOCK_READ8
#undef UNIT_BLOCK_READ8
#endif
#ifdef UNIT_BLOCK_WRITE
#undef UNIT_BLOCK_WRITE
#endif
#ifdef UNIT_BLOCK_WRITE2
#undef UNIT_BLOCK_WRITE2
#endif
#ifdef UNIT_BLOCK_WRITE4
#undef UNIT_BLOCK_WRITE4
#endif
#ifdef UNIT_BLOCK_WRITE8
#undef UNIT_BLOCK_WRITE8
#endif
#undef KERNEL
#undef KERNEL_ID
#undef FUNC
#undef FUNC_CALL
#undef CONST_ARRAY_DECL
#undef CONST_ARRAY_REF
#ifdef FP64_SUPPORTED
#undef FP64_SUPPORTED
#endif
#ifdef FP16_SUPPORTED
#undef FP16_SUPPORTED
#endif
#ifdef FP16_UNIT_USED
#undef FP16_UNIT_USED
#endif
#ifdef INT8_UNIT_USED
#undef INT8_UNIT_USED
#endif
#ifdef INT32_UNIT_USED
#undef INT32_UNIT_USED
#endif
#ifdef INT64_UNIT_USED
#undef INT64_UNIT_USED
#endif
#ifdef UINT8_UNIT_USED
#undef UINT8_UNIT_USED
#endif
#ifdef UINT32_UNIT_USED
#undef UINT32_UNIT_USED
#endif
#ifdef UNIT_TYPE
#undef UNIT_TYPE
#endif
#ifdef UNIT_VAL_MAX
#undef UNIT_VAL_MAX
#endif
#ifdef UNIT_VAL_MIN
#undef UNIT_VAL_MIN
#endif
#ifdef UNIT_VAL_ONE
#undef UNIT_VAL_ONE
#endif
#ifdef UNIT_VAL_ZERO
#undef UNIT_VAL_ZERO
#endif
#ifdef TO_UNIT_TYPE
#undef TO_UNIT_TYPE
#endif
#ifdef TO_UNIT_TYPE_SAT
#undef TO_UNIT_TYPE_SAT
#endif
#ifdef AS_UNIT_TYPE
#undef AS_UNIT_TYPE
#endif
#ifdef UNIT_MAX_FUNC
#undef UNIT_MAX_FUNC
#endif
#ifdef UNIT_MIN_FUNC
#undef UNIT_MIN_FUNC
#endif
#ifdef UNIT_ABS_FUNC
#undef UNIT_ABS_FUNC
#endif
#ifdef UNIT_TYPE_SIZE
#undef UNIT_TYPE_SIZE
#endif
#ifdef UNIT_IS_FP
#undef UNIT_IS_FP
#endif
#ifdef NL_M
#undef NL_M
#endif
#ifdef NL_N
#undef NL_N
#endif
#ifdef ACTIVATION_FUNC_TYPE
#undef ACTIVATION_FUNC_TYPE
#endif
#ifdef ACTIVATION_FUNC_VAL_MAX
#undef ACTIVATION_FUNC_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_VAL_MIN
#undef ACTIVATION_FUNC_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_VAL_ONE
#undef ACTIVATION_FUNC_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_VAL_ZERO
#undef ACTIVATION_FUNC_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_TYPE
#undef TO_ACTIVATION_FUNC_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_TYPE_SAT
#undef TO_ACTIVATION_FUNC_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_TYPE
#undef AS_ACTIVATION_FUNC_TYPE
#endif
#ifdef ACTIVATION_FUNC_MAX_FUNC
#undef ACTIVATION_FUNC_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_MIN_FUNC
#undef ACTIVATION_FUNC_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_ABS_FUNC
#undef ACTIVATION_FUNC_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_TYPE_SIZE
#undef ACTIVATION_FUNC_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_IS_FP
#undef ACTIVATION_FUNC_IS_FP
#endif
#ifdef ACTIVATION_PARAMS
#undef ACTIVATION_PARAMS
#endif
#ifdef ACTIVATION_FUNC
#undef ACTIVATION_FUNC
#endif
#ifdef ACTIVATION
#undef ACTIVATION
#endif
#ifdef INPUT0_SIZE_X
#undef INPUT0_SIZE_X
#endif
#ifdef INPUT0_SIZE_Y
#undef INPUT0_SIZE_Y
#endif
#ifdef INPUT0_SIZE_Z
#undef INPUT0_SIZE_Z
#endif
#ifdef INPUT0_SIZE_W
#undef INPUT0_SIZE_W
#endif
#ifdef INPUT0_SIZE_U
#undef INPUT0_SIZE_U
#endif
#ifdef INPUT0_SIZE_V
#undef INPUT0_SIZE_V
#endif
#ifdef INPUT0_FEATURE_NUM
#undef INPUT0_FEATURE_NUM
#endif
#ifdef INPUT0_BATCH_NUM
#undef INPUT0_BATCH_NUM
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_X
#undef INPUT0_PAD_BEFORE_SIZE_X
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_Y
#undef INPUT0_PAD_BEFORE_SIZE_Y
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_Z
#undef INPUT0_PAD_BEFORE_SIZE_Z
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_W
#undef INPUT0_PAD_BEFORE_SIZE_W
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_U
#undef INPUT0_PAD_BEFORE_SIZE_U
#endif
#ifdef INPUT0_PAD_BEFORE_SIZE_V
#undef INPUT0_PAD_BEFORE_SIZE_V
#endif
#ifdef INPUT0_PAD_BEFORE_FEATURE_NUM
#undef INPUT0_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef INPUT0_PAD_BEFORE_BATCH_NUM
#undef INPUT0_PAD_BEFORE_BATCH_NUM
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_X
#undef INPUT0_PAD_AFTER_SIZE_X
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_Y
#undef INPUT0_PAD_AFTER_SIZE_Y
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_Z
#undef INPUT0_PAD_AFTER_SIZE_Z
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_W
#undef INPUT0_PAD_AFTER_SIZE_W
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_U
#undef INPUT0_PAD_AFTER_SIZE_U
#endif
#ifdef INPUT0_PAD_AFTER_SIZE_V
#undef INPUT0_PAD_AFTER_SIZE_V
#endif
#ifdef INPUT0_PAD_AFTER_FEATURE_NUM
#undef INPUT0_PAD_AFTER_FEATURE_NUM
#endif
#ifdef INPUT0_PAD_AFTER_BATCH_NUM
#undef INPUT0_PAD_AFTER_BATCH_NUM
#endif
#ifdef INPUT0_LENGTH
#undef INPUT0_LENGTH
#endif
#ifdef INPUT0_X_PITCH
#undef INPUT0_X_PITCH
#endif
#ifdef INPUT0_Y_PITCH
#undef INPUT0_Y_PITCH
#endif
#ifdef INPUT0_Z_PITCH
#undef INPUT0_Z_PITCH
#endif
#ifdef INPUT0_W_PITCH
#undef INPUT0_W_PITCH
#endif
#ifdef INPUT0_U_PITCH
#undef INPUT0_U_PITCH
#endif
#ifdef INPUT0_V_PITCH
#undef INPUT0_V_PITCH
#endif
#ifdef INPUT0_FEATURE_PITCH
#undef INPUT0_FEATURE_PITCH
#endif
#ifdef INPUT0_BATCH_PITCH
#undef INPUT0_BATCH_PITCH
#endif
#ifdef INPUT0_GET_INDEX_SAFE
#undef INPUT0_GET_INDEX_SAFE
#endif
#ifdef INPUT0_GET_INDEX
#undef INPUT0_GET_INDEX
#endif
#ifdef INPUT0_GET_INDEX_RAW
#undef INPUT0_GET_INDEX_RAW
#endif
#ifdef INPUT0_VIEW_OFFSET
#undef INPUT0_VIEW_OFFSET
#endif
#ifdef INPUT0_DIMS
#undef INPUT0_DIMS
#endif
#ifdef INPUT0_SIMPLE
#undef INPUT0_SIMPLE
#endif
#ifdef INPUT0_GROUPED
#undef INPUT0_GROUPED
#endif
#ifdef INPUT0_LAYOUT_B_FS_YX_FSV16
#undef INPUT0_LAYOUT_B_FS_YX_FSV16
#endif
#ifdef INPUT0_TYPE
#undef INPUT0_TYPE
#endif
#ifdef INPUT0_VAL_MAX
#undef INPUT0_VAL_MAX
#endif
#ifdef INPUT0_VAL_MIN
#undef INPUT0_VAL_MIN
#endif
#ifdef INPUT0_VAL_ONE
#undef INPUT0_VAL_ONE
#endif
#ifdef INPUT0_VAL_ZERO
#undef INPUT0_VAL_ZERO
#endif
#ifdef TO_INPUT0_TYPE
#undef TO_INPUT0_TYPE
#endif
#ifdef TO_INPUT0_TYPE_SAT
#undef TO_INPUT0_TYPE_SAT
#endif
#ifdef AS_INPUT0_TYPE
#undef AS_INPUT0_TYPE
#endif
#ifdef INPUT0_MAX_FUNC
#undef INPUT0_MAX_FUNC
#endif
#ifdef INPUT0_MIN_FUNC
#undef INPUT0_MIN_FUNC
#endif
#ifdef INPUT0_ABS_FUNC
#undef INPUT0_ABS_FUNC
#endif
#ifdef INPUT0_TYPE_SIZE
#undef INPUT0_TYPE_SIZE
#endif
#ifdef INPUT0_IS_FP
#undef INPUT0_IS_FP
#endif
#ifdef INPUT0_OFFSET
#undef INPUT0_OFFSET
#endif
#ifdef INPUT0_SIZES_DATA
#undef INPUT0_SIZES_DATA
#endif
#ifdef INPUT0_PITCHES
#undef INPUT0_PITCHES
#endif
#ifdef INPUT0_PAD_BEFORE
#undef INPUT0_PAD_BEFORE
#endif
#ifdef INPUT0_PAD_AFTER
#undef INPUT0_PAD_AFTER
#endif
#ifdef OUTPUT_SIZE_X
#undef OUTPUT_SIZE_X
#endif
#ifdef OUTPUT_SIZE_Y
#undef OUTPUT_SIZE_Y
#endif
#ifdef OUTPUT_SIZE_Z
#undef OUTPUT_SIZE_Z
#endif
#ifdef OUTPUT_SIZE_W
#undef OUTPUT_SIZE_W
#endif
#ifdef OUTPUT_SIZE_U
#undef OUTPUT_SIZE_U
#endif
#ifdef OUTPUT_SIZE_V
#undef OUTPUT_SIZE_V
#endif
#ifdef OUTPUT_FEATURE_NUM
#undef OUTPUT_FEATURE_NUM
#endif
#ifdef OUTPUT_BATCH_NUM
#undef OUTPUT_BATCH_NUM
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_X
#undef OUTPUT_PAD_BEFORE_SIZE_X
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_Y
#undef OUTPUT_PAD_BEFORE_SIZE_Y
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_Z
#undef OUTPUT_PAD_BEFORE_SIZE_Z
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_W
#undef OUTPUT_PAD_BEFORE_SIZE_W
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_U
#undef OUTPUT_PAD_BEFORE_SIZE_U
#endif
#ifdef OUTPUT_PAD_BEFORE_SIZE_V
#undef OUTPUT_PAD_BEFORE_SIZE_V
#endif
#ifdef OUTPUT_PAD_BEFORE_FEATURE_NUM
#undef OUTPUT_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef OUTPUT_PAD_BEFORE_BATCH_NUM
#undef OUTPUT_PAD_BEFORE_BATCH_NUM
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_X
#undef OUTPUT_PAD_AFTER_SIZE_X
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_Y
#undef OUTPUT_PAD_AFTER_SIZE_Y
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_Z
#undef OUTPUT_PAD_AFTER_SIZE_Z
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_W
#undef OUTPUT_PAD_AFTER_SIZE_W
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_U
#undef OUTPUT_PAD_AFTER_SIZE_U
#endif
#ifdef OUTPUT_PAD_AFTER_SIZE_V
#undef OUTPUT_PAD_AFTER_SIZE_V
#endif
#ifdef OUTPUT_PAD_AFTER_FEATURE_NUM
#undef OUTPUT_PAD_AFTER_FEATURE_NUM
#endif
#ifdef OUTPUT_PAD_AFTER_BATCH_NUM
#undef OUTPUT_PAD_AFTER_BATCH_NUM
#endif
#ifdef OUTPUT_LENGTH
#undef OUTPUT_LENGTH
#endif
#ifdef OUTPUT_X_PITCH
#undef OUTPUT_X_PITCH
#endif
#ifdef OUTPUT_Y_PITCH
#undef OUTPUT_Y_PITCH
#endif
#ifdef OUTPUT_Z_PITCH
#undef OUTPUT_Z_PITCH
#endif
#ifdef OUTPUT_W_PITCH
#undef OUTPUT_W_PITCH
#endif
#ifdef OUTPUT_U_PITCH
#undef OUTPUT_U_PITCH
#endif
#ifdef OUTPUT_V_PITCH
#undef OUTPUT_V_PITCH
#endif
#ifdef OUTPUT_FEATURE_PITCH
#undef OUTPUT_FEATURE_PITCH
#endif
#ifdef OUTPUT_BATCH_PITCH
#undef OUTPUT_BATCH_PITCH
#endif
#ifdef OUTPUT_GET_INDEX_SAFE
#undef OUTPUT_GET_INDEX_SAFE
#endif
#ifdef OUTPUT_GET_INDEX
#undef OUTPUT_GET_INDEX
#endif
#ifdef OUTPUT_GET_INDEX_RAW
#undef OUTPUT_GET_INDEX_RAW
#endif
#ifdef OUTPUT_VIEW_OFFSET
#undef OUTPUT_VIEW_OFFSET
#endif
#ifdef OUTPUT_DIMS
#undef OUTPUT_DIMS
#endif
#ifdef OUTPUT_SIMPLE
#undef OUTPUT_SIMPLE
#endif
#ifdef OUTPUT_GROUPED
#undef OUTPUT_GROUPED
#endif
#ifdef OUTPUT_LAYOUT_B_FS_YX_FSV16
#undef OUTPUT_LAYOUT_B_FS_YX_FSV16
#endif
#ifdef OUTPUT_TYPE
#undef OUTPUT_TYPE
#endif
#ifdef OUTPUT_VAL_MAX
#undef OUTPUT_VAL_MAX
#endif
#ifdef OUTPUT_VAL_MIN
#undef OUTPUT_VAL_MIN
#endif
#ifdef OUTPUT_VAL_ONE
#undef OUTPUT_VAL_ONE
#endif
#ifdef OUTPUT_VAL_ZERO
#undef OUTPUT_VAL_ZERO
#endif
#ifdef TO_OUTPUT_TYPE
#undef TO_OUTPUT_TYPE
#endif
#ifdef TO_OUTPUT_TYPE_SAT
#undef TO_OUTPUT_TYPE_SAT
#endif
#ifdef AS_OUTPUT_TYPE
#undef AS_OUTPUT_TYPE
#endif
#ifdef OUTPUT_MAX_FUNC
#undef OUTPUT_MAX_FUNC
#endif
#ifdef OUTPUT_MIN_FUNC
#undef OUTPUT_MIN_FUNC
#endif
#ifdef OUTPUT_ABS_FUNC
#undef OUTPUT_ABS_FUNC
#endif
#ifdef OUTPUT_TYPE_SIZE
#undef OUTPUT_TYPE_SIZE
#endif
#ifdef OUTPUT_IS_FP
#undef OUTPUT_IS_FP
#endif
#ifdef OUTPUT_OFFSET
#undef OUTPUT_OFFSET
#endif
#ifdef OUTPUT_SIZES_DATA
#undef OUTPUT_SIZES_DATA
#endif
#ifdef OUTPUT_PITCHES
#undef OUTPUT_PITCHES
#endif
#ifdef OUTPUT_PAD_BEFORE
#undef OUTPUT_PAD_BEFORE
#endif
#ifdef OUTPUT_PAD_AFTER
#undef OUTPUT_PAD_AFTER
#endif
#ifdef OPTIONAL_SHAPE_INFO_ARG
#undef OPTIONAL_SHAPE_INFO_ARG
#endif
#ifdef OPTIONAL_SHAPE_INFO_TENSOR
#undef OPTIONAL_SHAPE_INFO_TENSOR
#endif
#ifdef FILTER_SIZE_X
#undef FILTER_SIZE_X
#endif
#ifdef FILTER_SIZE_Y
#undef FILTER_SIZE_Y
#endif
#ifdef FILTER_SIZE_Z
#undef FILTER_SIZE_Z
#endif
#ifdef FILTER_IFM_NUM
#undef FILTER_IFM_NUM
#endif
#ifdef FILTER_OFM_NUM
#undef FILTER_OFM_NUM
#endif
#ifdef FILTER_GROUPS_NUM
#undef FILTER_GROUPS_NUM
#endif
#ifdef FILTER_X_PITCH
#undef FILTER_X_PITCH
#endif
#ifdef FILTER_Y_PITCH
#undef FILTER_Y_PITCH
#endif
#ifdef FILTER_Z_PITCH
#undef FILTER_Z_PITCH
#endif
#ifdef FILTER_IFM_PITCH
#undef FILTER_IFM_PITCH
#endif
#ifdef FILTER_OFM_PITCH
#undef FILTER_OFM_PITCH
#endif
#ifdef FILTER_GROUPS_PITCH
#undef FILTER_GROUPS_PITCH
#endif
#ifdef FILTER_LENGTH
#undef FILTER_LENGTH
#endif
#ifdef FILTER_VIEW_OFFSET
#undef FILTER_VIEW_OFFSET
#endif
#ifdef FILTER_DIMS
#undef FILTER_DIMS
#endif
#ifdef FILTER_SIMPLE
#undef FILTER_SIMPLE
#endif
#ifdef FILTER_GROUPED
#undef FILTER_GROUPED
#endif
#ifdef FILTER_LAYOUT_OS_IS_YX_ISV16_OSV16
#undef FILTER_LAYOUT_OS_IS_YX_ISV16_OSV16
#endif
#ifdef FILTER_TYPE
#undef FILTER_TYPE
#endif
#ifdef FILTER_VAL_MAX
#undef FILTER_VAL_MAX
#endif
#ifdef FILTER_VAL_MIN
#undef FILTER_VAL_MIN
#endif
#ifdef FILTER_VAL_ONE
#undef FILTER_VAL_ONE
#endif
#ifdef FILTER_VAL_ZERO
#undef FILTER_VAL_ZERO
#endif
#ifdef TO_FILTER_TYPE
#undef TO_FILTER_TYPE
#endif
#ifdef TO_FILTER_TYPE_SAT
#undef TO_FILTER_TYPE_SAT
#endif
#ifdef AS_FILTER_TYPE
#undef AS_FILTER_TYPE
#endif
#ifdef FILTER_MAX_FUNC
#undef FILTER_MAX_FUNC
#endif
#ifdef FILTER_MIN_FUNC
#undef FILTER_MIN_FUNC
#endif
#ifdef FILTER_ABS_FUNC
#undef FILTER_ABS_FUNC
#endif
#ifdef FILTER_TYPE_SIZE
#undef FILTER_TYPE_SIZE
#endif
#ifdef FILTER_IS_FP
#undef FILTER_IS_FP
#endif
#ifdef FILTER_OFFSET
#undef FILTER_OFFSET
#endif
#ifdef FILTER_SIZES_DATA
#undef FILTER_SIZES_DATA
#endif
#ifdef FILTER_PITCHES
#undef FILTER_PITCHES
#endif
#ifdef FILTER_PAD_BEFORE
#undef FILTER_PAD_BEFORE
#endif
#ifdef FILTER_PAD_AFTER
#undef FILTER_PAD_AFTER
#endif
#ifdef BIAS_TERM
#undef BIAS_TERM
#endif
#ifdef BIAS_SIZE_X
#undef BIAS_SIZE_X
#endif
#ifdef BIAS_SIZE_Y
#undef BIAS_SIZE_Y
#endif
#ifdef BIAS_SIZE_Z
#undef BIAS_SIZE_Z
#endif
#ifdef BIAS_SIZE_W
#undef BIAS_SIZE_W
#endif
#ifdef BIAS_SIZE_U
#undef BIAS_SIZE_U
#endif
#ifdef BIAS_SIZE_V
#undef BIAS_SIZE_V
#endif
#ifdef BIAS_FEATURE_NUM
#undef BIAS_FEATURE_NUM
#endif
#ifdef BIAS_BATCH_NUM
#undef BIAS_BATCH_NUM
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_X
#undef BIAS_PAD_BEFORE_SIZE_X
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_Y
#undef BIAS_PAD_BEFORE_SIZE_Y
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_Z
#undef BIAS_PAD_BEFORE_SIZE_Z
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_W
#undef BIAS_PAD_BEFORE_SIZE_W
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_U
#undef BIAS_PAD_BEFORE_SIZE_U
#endif
#ifdef BIAS_PAD_BEFORE_SIZE_V
#undef BIAS_PAD_BEFORE_SIZE_V
#endif
#ifdef BIAS_PAD_BEFORE_FEATURE_NUM
#undef BIAS_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef BIAS_PAD_BEFORE_BATCH_NUM
#undef BIAS_PAD_BEFORE_BATCH_NUM
#endif
#ifdef BIAS_PAD_AFTER_SIZE_X
#undef BIAS_PAD_AFTER_SIZE_X
#endif
#ifdef BIAS_PAD_AFTER_SIZE_Y
#undef BIAS_PAD_AFTER_SIZE_Y
#endif
#ifdef BIAS_PAD_AFTER_SIZE_Z
#undef BIAS_PAD_AFTER_SIZE_Z
#endif
#ifdef BIAS_PAD_AFTER_SIZE_W
#undef BIAS_PAD_AFTER_SIZE_W
#endif
#ifdef BIAS_PAD_AFTER_SIZE_U
#undef BIAS_PAD_AFTER_SIZE_U
#endif
#ifdef BIAS_PAD_AFTER_SIZE_V
#undef BIAS_PAD_AFTER_SIZE_V
#endif
#ifdef BIAS_PAD_AFTER_FEATURE_NUM
#undef BIAS_PAD_AFTER_FEATURE_NUM
#endif
#ifdef BIAS_PAD_AFTER_BATCH_NUM
#undef BIAS_PAD_AFTER_BATCH_NUM
#endif
#ifdef BIAS_LENGTH
#undef BIAS_LENGTH
#endif
#ifdef BIAS_X_PITCH
#undef BIAS_X_PITCH
#endif
#ifdef BIAS_Y_PITCH
#undef BIAS_Y_PITCH
#endif
#ifdef BIAS_Z_PITCH
#undef BIAS_Z_PITCH
#endif
#ifdef BIAS_W_PITCH
#undef BIAS_W_PITCH
#endif
#ifdef BIAS_U_PITCH
#undef BIAS_U_PITCH
#endif
#ifdef BIAS_V_PITCH
#undef BIAS_V_PITCH
#endif
#ifdef BIAS_FEATURE_PITCH
#undef BIAS_FEATURE_PITCH
#endif
#ifdef BIAS_BATCH_PITCH
#undef BIAS_BATCH_PITCH
#endif
#ifdef BIAS_GET_INDEX_SAFE
#undef BIAS_GET_INDEX_SAFE
#endif
#ifdef BIAS_GET_INDEX
#undef BIAS_GET_INDEX
#endif
#ifdef BIAS_GET_INDEX_RAW
#undef BIAS_GET_INDEX_RAW
#endif
#ifdef BIAS_VIEW_OFFSET
#undef BIAS_VIEW_OFFSET
#endif
#ifdef BIAS_DIMS
#undef BIAS_DIMS
#endif
#ifdef BIAS_SIMPLE
#undef BIAS_SIMPLE
#endif
#ifdef BIAS_GROUPED
#undef BIAS_GROUPED
#endif
#ifdef BIAS_LAYOUT_BF
#undef BIAS_LAYOUT_BF
#endif
#ifdef BIAS_TYPE
#undef BIAS_TYPE
#endif
#ifdef BIAS_VAL_MAX
#undef BIAS_VAL_MAX
#endif
#ifdef BIAS_VAL_MIN
#undef BIAS_VAL_MIN
#endif
#ifdef BIAS_VAL_ONE
#undef BIAS_VAL_ONE
#endif
#ifdef BIAS_VAL_ZERO
#undef BIAS_VAL_ZERO
#endif
#ifdef TO_BIAS_TYPE
#undef TO_BIAS_TYPE
#endif
#ifdef TO_BIAS_TYPE_SAT
#undef TO_BIAS_TYPE_SAT
#endif
#ifdef AS_BIAS_TYPE
#undef AS_BIAS_TYPE
#endif
#ifdef BIAS_MAX_FUNC
#undef BIAS_MAX_FUNC
#endif
#ifdef BIAS_MIN_FUNC
#undef BIAS_MIN_FUNC
#endif
#ifdef BIAS_ABS_FUNC
#undef BIAS_ABS_FUNC
#endif
#ifdef BIAS_TYPE_SIZE
#undef BIAS_TYPE_SIZE
#endif
#ifdef BIAS_IS_FP
#undef BIAS_IS_FP
#endif
#ifdef BIAS_OFFSET
#undef BIAS_OFFSET
#endif
#ifdef BIAS_SIZES_DATA
#undef BIAS_SIZES_DATA
#endif
#ifdef BIAS_PITCHES
#undef BIAS_PITCHES
#endif
#ifdef BIAS_PAD_BEFORE
#undef BIAS_PAD_BEFORE
#endif
#ifdef BIAS_PAD_AFTER
#undef BIAS_PAD_AFTER
#endif
#ifdef BIAS_PER_OUTPUT
#undef BIAS_PER_OUTPUT
#endif
#ifdef BIAS_PER_OFM
#undef BIAS_PER_OFM
#endif
#ifdef STRIDE_SIZE_X
#undef STRIDE_SIZE_X
#endif
#ifdef STRIDE_SIZE_Y
#undef STRIDE_SIZE_Y
#endif
#ifdef STRIDE_SIZE_Z
#undef STRIDE_SIZE_Z
#endif
#ifdef PADDING_SIZE_X
#undef PADDING_SIZE_X
#endif
#ifdef PADDING_SIZE_Y
#undef PADDING_SIZE_Y
#endif
#ifdef PADDING_SIZE_Z
#undef PADDING_SIZE_Z
#endif
#ifdef DILATION_SIZE_X
#undef DILATION_SIZE_X
#endif
#ifdef DILATION_SIZE_Y
#undef DILATION_SIZE_Y
#endif
#ifdef DILATION_SIZE_Z
#undef DILATION_SIZE_Z
#endif
#ifdef FILTER_ARRAY_NUM
#undef FILTER_ARRAY_NUM
#endif
#ifdef INPUT0_OFFSET_WITH_PADDING
#undef INPUT0_OFFSET_WITH_PADDING
#endif
#ifdef GROUPED
#undef GROUPED
#endif
#ifdef SKIP_BATCH
#undef SKIP_BATCH
#endif
#ifdef FUSED_OP0_LOAD_VEC
#undef FUSED_OP0_LOAD_VEC
#endif
#ifdef FUSED_OP0_ACTION_VEC
#undef FUSED_OP0_ACTION_VEC
#endif
#ifdef FUSED_OP1_LOAD_VEC
#undef FUSED_OP1_LOAD_VEC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_MAX
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_MIN
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_ONE
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_ZERO
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE
#undef TO_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE_SAT
#undef TO_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE
#undef AS_ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_MAX_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_MIN_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_ABS_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE_SIZE
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC_IS_FP
#undef ACTIVATION_FUNC_FUSED_OP1_VEC_IS_FP
#endif
#ifdef ACTIVATION_PARAMS_FUSED_OP1_VEC
#undef ACTIVATION_PARAMS_FUSED_OP1_VEC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_VEC
#undef ACTIVATION_FUNC_FUSED_OP1_VEC
#endif
#ifdef ACTIVATION_FUSED_OP1_VEC
#undef ACTIVATION_FUSED_OP1_VEC
#endif
#ifdef FUSED_OP1_ACTION_VEC
#undef FUSED_OP1_ACTION_VEC
#endif
#ifdef FUSED_OPS_VEC
#undef FUSED_OPS_VEC
#endif
#ifdef FUSED_OPS_PRELOAD_VEC
#undef FUSED_OPS_PRELOAD_VEC
#endif
#ifdef FUSED_OPS_CALC_VEC
#undef FUSED_OPS_CALC_VEC
#endif
#ifdef FUSED_OPS_RESULT_VEC
#undef FUSED_OPS_RESULT_VEC
#endif
#ifdef FUSED_OPS_CAN_USE_PRELOAD_VEC
#undef FUSED_OPS_CAN_USE_PRELOAD_VEC
#endif
#ifdef FUSED_OP0_LOAD_SCALAR
#undef FUSED_OP0_LOAD_SCALAR
#endif
#ifdef FUSED_OP0_ACTION_SCALAR
#undef FUSED_OP0_ACTION_SCALAR
#endif
#ifdef FUSED_OP1_LOAD_SCALAR
#undef FUSED_OP1_LOAD_SCALAR
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_MAX
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_MIN
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_ONE
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_ZERO
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE
#undef TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE_SAT
#undef TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE
#undef AS_ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_MAX_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_MIN_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_ABS_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE_SIZE
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_IS_FP
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_IS_FP
#endif
#ifdef ACTIVATION_PARAMS_FUSED_OP1_SCALAR
#undef ACTIVATION_PARAMS_FUSED_OP1_SCALAR
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR
#endif
#ifdef ACTIVATION_FUSED_OP1_SCALAR
#undef ACTIVATION_FUSED_OP1_SCALAR
#endif
#ifdef FUSED_OP1_ACTION_SCALAR
#undef FUSED_OP1_ACTION_SCALAR
#endif
#ifdef FUSED_OPS_SCALAR
#undef FUSED_OPS_SCALAR
#endif
#ifdef FUSED_OPS_PRELOAD_SCALAR
#undef FUSED_OPS_PRELOAD_SCALAR
#endif
#ifdef FUSED_OPS_CALC_SCALAR
#undef FUSED_OPS_CALC_SCALAR
#endif
#ifdef FUSED_OPS_RESULT_SCALAR
#undef FUSED_OPS_RESULT_SCALAR
#endif
#ifdef FUSED_OPS_CAN_USE_PRELOAD_SCALAR
#undef FUSED_OPS_CAN_USE_PRELOAD_SCALAR
#endif
#ifdef FUSED_OP0_LOAD_SCALAR_B1
#undef FUSED_OP0_LOAD_SCALAR_B1
#endif
#ifdef FUSED_OP0_ACTION_SCALAR_B1
#undef FUSED_OP0_ACTION_SCALAR_B1
#endif
#ifdef FUSED_OP1_LOAD_SCALAR_B1
#undef FUSED_OP1_LOAD_SCALAR_B1
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_MAX
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_MAX
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_MIN
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_MIN
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_ONE
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_ONE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_ZERO
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_VAL_ZERO
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE
#undef TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE
#endif
#ifdef TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE_SAT
#undef TO_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE_SAT
#endif
#ifdef AS_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE
#undef AS_ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_MAX_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_MAX_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_MIN_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_MIN_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_ABS_FUNC
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_ABS_FUNC
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE_SIZE
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_TYPE_SIZE
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_IS_FP
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1_IS_FP
#endif
#ifdef ACTIVATION_PARAMS_FUSED_OP1_SCALAR_B1
#undef ACTIVATION_PARAMS_FUSED_OP1_SCALAR_B1
#endif
#ifdef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1
#undef ACTIVATION_FUNC_FUSED_OP1_SCALAR_B1
#endif
#ifdef ACTIVATION_FUSED_OP1_SCALAR_B1
#undef ACTIVATION_FUSED_OP1_SCALAR_B1
#endif
#ifdef FUSED_OP1_ACTION_SCALAR_B1
#undef FUSED_OP1_ACTION_SCALAR_B1
#endif
#ifdef FUSED_OPS_SCALAR_B1
#undef FUSED_OPS_SCALAR_B1
#endif
#ifdef FUSED_OPS_PRELOAD_SCALAR_B1
#undef FUSED_OPS_PRELOAD_SCALAR_B1
#endif
#ifdef FUSED_OPS_CALC_SCALAR_B1
#undef FUSED_OPS_CALC_SCALAR_B1
#endif
#ifdef FUSED_OPS_RESULT_SCALAR_B1
#undef FUSED_OPS_RESULT_SCALAR_B1
#endif
#ifdef FUSED_OPS_CAN_USE_PRELOAD_SCALAR_B1
#undef FUSED_OPS_CAN_USE_PRELOAD_SCALAR_B1
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_X
#undef FUSED_OP_0_INPUT0_SIZE_X
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_Y
#undef FUSED_OP_0_INPUT0_SIZE_Y
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_Z
#undef FUSED_OP_0_INPUT0_SIZE_Z
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_W
#undef FUSED_OP_0_INPUT0_SIZE_W
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_U
#undef FUSED_OP_0_INPUT0_SIZE_U
#endif
#ifdef FUSED_OP_0_INPUT0_SIZE_V
#undef FUSED_OP_0_INPUT0_SIZE_V
#endif
#ifdef FUSED_OP_0_INPUT0_FEATURE_NUM
#undef FUSED_OP_0_INPUT0_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_BATCH_NUM
#undef FUSED_OP_0_INPUT0_BATCH_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_X
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_X
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Y
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Y
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Z
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_Z
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_W
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_W
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_U
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_U
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_V
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_SIZE_V
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_FEATURE_NUM
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE_BATCH_NUM
#undef FUSED_OP_0_INPUT0_PAD_BEFORE_BATCH_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_X
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_X
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Y
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Y
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Z
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_Z
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_W
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_W
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_U
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_U
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_V
#undef FUSED_OP_0_INPUT0_PAD_AFTER_SIZE_V
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_FEATURE_NUM
#undef FUSED_OP_0_INPUT0_PAD_AFTER_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER_BATCH_NUM
#undef FUSED_OP_0_INPUT0_PAD_AFTER_BATCH_NUM
#endif
#ifdef FUSED_OP_0_INPUT0_LENGTH
#undef FUSED_OP_0_INPUT0_LENGTH
#endif
#ifdef FUSED_OP_0_INPUT0_X_PITCH
#undef FUSED_OP_0_INPUT0_X_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_Y_PITCH
#undef FUSED_OP_0_INPUT0_Y_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_Z_PITCH
#undef FUSED_OP_0_INPUT0_Z_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_W_PITCH
#undef FUSED_OP_0_INPUT0_W_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_U_PITCH
#undef FUSED_OP_0_INPUT0_U_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_V_PITCH
#undef FUSED_OP_0_INPUT0_V_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_FEATURE_PITCH
#undef FUSED_OP_0_INPUT0_FEATURE_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_BATCH_PITCH
#undef FUSED_OP_0_INPUT0_BATCH_PITCH
#endif
#ifdef FUSED_OP_0_INPUT0_GET_INDEX_SAFE
#undef FUSED_OP_0_INPUT0_GET_INDEX_SAFE
#endif
#ifdef FUSED_OP_0_INPUT0_GET_INDEX
#undef FUSED_OP_0_INPUT0_GET_INDEX
#endif
#ifdef FUSED_OP_0_INPUT0_GET_INDEX_RAW
#undef FUSED_OP_0_INPUT0_GET_INDEX_RAW
#endif
#ifdef FUSED_OP_0_INPUT0_VIEW_OFFSET
#undef FUSED_OP_0_INPUT0_VIEW_OFFSET
#endif
#ifdef FUSED_OP_0_INPUT0_DIMS
#undef FUSED_OP_0_INPUT0_DIMS
#endif
#ifdef FUSED_OP_0_INPUT0_SIMPLE
#undef FUSED_OP_0_INPUT0_SIMPLE
#endif
#ifdef FUSED_OP_0_INPUT0_GROUPED
#undef FUSED_OP_0_INPUT0_GROUPED
#endif
#ifdef FUSED_OP_0_INPUT0_LAYOUT_B_FS_YX_FSV16
#undef FUSED_OP_0_INPUT0_LAYOUT_B_FS_YX_FSV16
#endif
#ifdef FUSED_OP_0_INPUT0_TYPE
#undef FUSED_OP_0_INPUT0_TYPE
#endif
#ifdef FUSED_OP_0_INPUT0_VAL_MAX
#undef FUSED_OP_0_INPUT0_VAL_MAX
#endif
#ifdef FUSED_OP_0_INPUT0_VAL_MIN
#undef FUSED_OP_0_INPUT0_VAL_MIN
#endif
#ifdef FUSED_OP_0_INPUT0_VAL_ONE
#undef FUSED_OP_0_INPUT0_VAL_ONE
#endif
#ifdef FUSED_OP_0_INPUT0_VAL_ZERO
#undef FUSED_OP_0_INPUT0_VAL_ZERO
#endif
#ifdef TO_FUSED_OP_0_INPUT0_TYPE
#undef TO_FUSED_OP_0_INPUT0_TYPE
#endif
#ifdef TO_FUSED_OP_0_INPUT0_TYPE_SAT
#undef TO_FUSED_OP_0_INPUT0_TYPE_SAT
#endif
#ifdef AS_FUSED_OP_0_INPUT0_TYPE
#undef AS_FUSED_OP_0_INPUT0_TYPE
#endif
#ifdef FUSED_OP_0_INPUT0_MAX_FUNC
#undef FUSED_OP_0_INPUT0_MAX_FUNC
#endif
#ifdef FUSED_OP_0_INPUT0_MIN_FUNC
#undef FUSED_OP_0_INPUT0_MIN_FUNC
#endif
#ifdef FUSED_OP_0_INPUT0_ABS_FUNC
#undef FUSED_OP_0_INPUT0_ABS_FUNC
#endif
#ifdef FUSED_OP_0_INPUT0_TYPE_SIZE
#undef FUSED_OP_0_INPUT0_TYPE_SIZE
#endif
#ifdef FUSED_OP_0_INPUT0_IS_FP
#undef FUSED_OP_0_INPUT0_IS_FP
#endif
#ifdef FUSED_OP_0_INPUT0_OFFSET
#undef FUSED_OP_0_INPUT0_OFFSET
#endif
#ifdef FUSED_OP_0_INPUT0_SIZES_DATA
#undef FUSED_OP_0_INPUT0_SIZES_DATA
#endif
#ifdef FUSED_OP_0_INPUT0_PITCHES
#undef FUSED_OP_0_INPUT0_PITCHES
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_BEFORE
#undef FUSED_OP_0_INPUT0_PAD_BEFORE
#endif
#ifdef FUSED_OP_0_INPUT0_PAD_AFTER
#undef FUSED_OP_0_INPUT0_PAD_AFTER
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_X
#undef FUSED_OP_0_OUTPUT_SIZE_X
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_Y
#undef FUSED_OP_0_OUTPUT_SIZE_Y
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_Z
#undef FUSED_OP_0_OUTPUT_SIZE_Z
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_W
#undef FUSED_OP_0_OUTPUT_SIZE_W
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_U
#undef FUSED_OP_0_OUTPUT_SIZE_U
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZE_V
#undef FUSED_OP_0_OUTPUT_SIZE_V
#endif
#ifdef FUSED_OP_0_OUTPUT_FEATURE_NUM
#undef FUSED_OP_0_OUTPUT_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_BATCH_NUM
#undef FUSED_OP_0_OUTPUT_BATCH_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_X
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_X
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Y
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Y
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Z
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_Z
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_W
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_W
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_U
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_U
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_V
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_SIZE_V
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_FEATURE_NUM
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE_BATCH_NUM
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE_BATCH_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_X
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_X
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Y
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Y
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Z
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_Z
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_W
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_W
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_U
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_U
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_V
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_SIZE_V
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_FEATURE_NUM
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_FEATURE_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER_BATCH_NUM
#undef FUSED_OP_0_OUTPUT_PAD_AFTER_BATCH_NUM
#endif
#ifdef FUSED_OP_0_OUTPUT_LENGTH
#undef FUSED_OP_0_OUTPUT_LENGTH
#endif
#ifdef FUSED_OP_0_OUTPUT_X_PITCH
#undef FUSED_OP_0_OUTPUT_X_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_Y_PITCH
#undef FUSED_OP_0_OUTPUT_Y_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_Z_PITCH
#undef FUSED_OP_0_OUTPUT_Z_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_W_PITCH
#undef FUSED_OP_0_OUTPUT_W_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_U_PITCH
#undef FUSED_OP_0_OUTPUT_U_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_V_PITCH
#undef FUSED_OP_0_OUTPUT_V_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_FEATURE_PITCH
#undef FUSED_OP_0_OUTPUT_FEATURE_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_BATCH_PITCH
#undef FUSED_OP_0_OUTPUT_BATCH_PITCH
#endif
#ifdef FUSED_OP_0_OUTPUT_GET_INDEX_SAFE
#undef FUSED_OP_0_OUTPUT_GET_INDEX_SAFE
#endif
#ifdef FUSED_OP_0_OUTPUT_GET_INDEX
#undef FUSED_OP_0_OUTPUT_GET_INDEX
#endif
#ifdef FUSED_OP_0_OUTPUT_GET_INDEX_RAW
#undef FUSED_OP_0_OUTPUT_GET_INDEX_RAW
#endif
#ifdef FUSED_OP_0_OUTPUT_VIEW_OFFSET
#undef FUSED_OP_0_OUTPUT_VIEW_OFFSET
#endif
#ifdef FUSED_OP_0_OUTPUT_DIMS
#undef FUSED_OP_0_OUTPUT_DIMS
#endif
#ifdef FUSED_OP_0_OUTPUT_SIMPLE
#undef FUSED_OP_0_OUTPUT_SIMPLE
#endif
#ifdef FUSED_OP_0_OUTPUT_GROUPED
#undef FUSED_OP_0_OUTPUT_GROUPED
#endif
#ifdef FUSED_OP_0_OUTPUT_LAYOUT_BFYX
#undef FUSED_OP_0_OUTPUT_LAYOUT_BFYX
#endif
#ifdef FUSED_OP_0_OUTPUT_TYPE
#undef FUSED_OP_0_OUTPUT_TYPE
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_MAX
#undef FUSED_OP_0_OUTPUT_VAL_MAX
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_MIN
#undef FUSED_OP_0_OUTPUT_VAL_MIN
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_ONE
#undef FUSED_OP_0_OUTPUT_VAL_ONE
#endif
#ifdef FUSED_OP_0_OUTPUT_VAL_ZERO
#undef FUSED_OP_0_OUTPUT_VAL_ZERO
#endif
#ifdef TO_FUSED_OP_0_OUTPUT_TYPE
#undef TO_FUSED_OP_0_OUTPUT_TYPE
#endif
#ifdef TO_FUSED_OP_0_OUTPUT_TYPE_SAT
#undef TO_FUSED_OP_0_OUTPUT_TYPE_SAT
#endif
#ifdef AS_FUSED_OP_0_OUTPUT_TYPE
#undef AS_FUSED_OP_0_OUTPUT_TYPE
#endif
#ifdef FUSED_OP_0_OUTPUT_MAX_FUNC
#undef FUSED_OP_0_OUTPUT_MAX_FUNC
#endif
#ifdef FUSED_OP_0_OUTPUT_MIN_FUNC
#undef FUSED_OP_0_OUTPUT_MIN_FUNC
#endif
#ifdef FUSED_OP_0_OUTPUT_ABS_FUNC
#undef FUSED_OP_0_OUTPUT_ABS_FUNC
#endif
#ifdef FUSED_OP_0_OUTPUT_TYPE_SIZE
#undef FUSED_OP_0_OUTPUT_TYPE_SIZE
#endif
#ifdef FUSED_OP_0_OUTPUT_IS_FP
#undef FUSED_OP_0_OUTPUT_IS_FP
#endif
#ifdef FUSED_OP_0_OUTPUT_OFFSET
#undef FUSED_OP_0_OUTPUT_OFFSET
#endif
#ifdef FUSED_OP_0_OUTPUT_SIZES_DATA
#undef FUSED_OP_0_OUTPUT_SIZES_DATA
#endif
#ifdef FUSED_OP_0_OUTPUT_PITCHES
#undef FUSED_OP_0_OUTPUT_PITCHES
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_BEFORE
#undef FUSED_OP_0_OUTPUT_PAD_BEFORE
#endif
#ifdef FUSED_OP_0_OUTPUT_PAD_AFTER
#undef FUSED_OP_0_OUTPUT_PAD_AFTER
#endif
#ifdef FUSED_OP0_DECLS
#undef FUSED_OP0_DECLS
#endif
#ifdef FUSED_OP0_ARGS
#undef FUSED_OP0_ARGS
#endif
#ifdef FUSED_OP_1_OUTPUT_SIZE_X
#undef FUSED_OP_1_OUTPUT_SIZE_X
#endif
#ifdef FUSED_OP_1_OUTPUT_SIZE_Y
#undef FUSED_OP_1_OUTPUT_SIZE_Y
#endif
#ifdef FUSED_OP_1_OUTPUT_SIZE_Z
#undef FUSED_OP_1_OUTPUT_SIZE_Z
#endif
#ifdef FUSED_OP_1_OUTPUT_SIZE_W
#undef FUSED_OP_1_OUTPUT_SIZE_W
#endif
#ifdef FUSED_OP_1_OUTPUT_SIZE_U
#undef FUSED_OP_1_OUTPUT_SIZE_U
#endif
#ifdef FUSED_OP_1_OUTPUT_SIZE_V
#undef FUSED_OP_1_OUTPUT_SIZE_V
#endif
#ifdef FUSED_OP_1_OUTPUT_FEATURE_NUM
#undef FUSED_OP_1_OUTPUT_FEATURE_NUM
#endif
#ifdef FUSED_OP_1_OUTPUT_BATCH_NUM
#undef FUSED_OP_1_OUTPUT_BATCH_NUM
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_X
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_X
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_Y
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_Y
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_Z
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_Z
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_W
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_W
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_U
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_U
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_V
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE_SIZE_V
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE_FEATURE_NUM
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE_FEATURE_NUM
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE_BATCH_NUM
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE_BATCH_NUM
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_X
#undef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_X
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_Y
#undef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_Y
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_Z
#undef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_Z
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_W
#undef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_W
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_U
#undef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_U
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_V
#undef FUSED_OP_1_OUTPUT_PAD_AFTER_SIZE_V
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER_FEATURE_NUM
#undef FUSED_OP_1_OUTPUT_PAD_AFTER_FEATURE_NUM
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER_BATCH_NUM
#undef FUSED_OP_1_OUTPUT_PAD_AFTER_BATCH_NUM
#endif
#ifdef FUSED_OP_1_OUTPUT_LENGTH
#undef FUSED_OP_1_OUTPUT_LENGTH
#endif
#ifdef FUSED_OP_1_OUTPUT_X_PITCH
#undef FUSED_OP_1_OUTPUT_X_PITCH
#endif
#ifdef FUSED_OP_1_OUTPUT_Y_PITCH
#undef FUSED_OP_1_OUTPUT_Y_PITCH
#endif
#ifdef FUSED_OP_1_OUTPUT_Z_PITCH
#undef FUSED_OP_1_OUTPUT_Z_PITCH
#endif
#ifdef FUSED_OP_1_OUTPUT_W_PITCH
#undef FUSED_OP_1_OUTPUT_W_PITCH
#endif
#ifdef FUSED_OP_1_OUTPUT_U_PITCH
#undef FUSED_OP_1_OUTPUT_U_PITCH
#endif
#ifdef FUSED_OP_1_OUTPUT_V_PITCH
#undef FUSED_OP_1_OUTPUT_V_PITCH
#endif
#ifdef FUSED_OP_1_OUTPUT_FEATURE_PITCH
#undef FUSED_OP_1_OUTPUT_FEATURE_PITCH
#endif
#ifdef FUSED_OP_1_OUTPUT_BATCH_PITCH
#undef FUSED_OP_1_OUTPUT_BATCH_PITCH
#endif
#ifdef FUSED_OP_1_OUTPUT_GET_INDEX_SAFE
#undef FUSED_OP_1_OUTPUT_GET_INDEX_SAFE
#endif
#ifdef FUSED_OP_1_OUTPUT_GET_INDEX
#undef FUSED_OP_1_OUTPUT_GET_INDEX
#endif
#ifdef FUSED_OP_1_OUTPUT_GET_INDEX_RAW
#undef FUSED_OP_1_OUTPUT_GET_INDEX_RAW
#endif
#ifdef FUSED_OP_1_OUTPUT_VIEW_OFFSET
#undef FUSED_OP_1_OUTPUT_VIEW_OFFSET
#endif
#ifdef FUSED_OP_1_OUTPUT_DIMS
#undef FUSED_OP_1_OUTPUT_DIMS
#endif
#ifdef FUSED_OP_1_OUTPUT_SIMPLE
#undef FUSED_OP_1_OUTPUT_SIMPLE
#endif
#ifdef FUSED_OP_1_OUTPUT_GROUPED
#undef FUSED_OP_1_OUTPUT_GROUPED
#endif
#ifdef FUSED_OP_1_OUTPUT_LAYOUT_BFYX
#undef FUSED_OP_1_OUTPUT_LAYOUT_BFYX
#endif
#ifdef FUSED_OP_1_OUTPUT_TYPE
#undef FUSED_OP_1_OUTPUT_TYPE
#endif
#ifdef FUSED_OP_1_OUTPUT_VAL_MAX
#undef FUSED_OP_1_OUTPUT_VAL_MAX
#endif
#ifdef FUSED_OP_1_OUTPUT_VAL_MIN
#undef FUSED_OP_1_OUTPUT_VAL_MIN
#endif
#ifdef FUSED_OP_1_OUTPUT_VAL_ONE
#undef FUSED_OP_1_OUTPUT_VAL_ONE
#endif
#ifdef FUSED_OP_1_OUTPUT_VAL_ZERO
#undef FUSED_OP_1_OUTPUT_VAL_ZERO
#endif
#ifdef TO_FUSED_OP_1_OUTPUT_TYPE
#undef TO_FUSED_OP_1_OUTPUT_TYPE
#endif
#ifdef TO_FUSED_OP_1_OUTPUT_TYPE_SAT
#undef TO_FUSED_OP_1_OUTPUT_TYPE_SAT
#endif
#ifdef AS_FUSED_OP_1_OUTPUT_TYPE
#undef AS_FUSED_OP_1_OUTPUT_TYPE
#endif
#ifdef FUSED_OP_1_OUTPUT_MAX_FUNC
#undef FUSED_OP_1_OUTPUT_MAX_FUNC
#endif
#ifdef FUSED_OP_1_OUTPUT_MIN_FUNC
#undef FUSED_OP_1_OUTPUT_MIN_FUNC
#endif
#ifdef FUSED_OP_1_OUTPUT_ABS_FUNC
#undef FUSED_OP_1_OUTPUT_ABS_FUNC
#endif
#ifdef FUSED_OP_1_OUTPUT_TYPE_SIZE
#undef FUSED_OP_1_OUTPUT_TYPE_SIZE
#endif
#ifdef FUSED_OP_1_OUTPUT_IS_FP
#undef FUSED_OP_1_OUTPUT_IS_FP
#endif
#ifdef FUSED_OP_1_OUTPUT_OFFSET
#undef FUSED_OP_1_OUTPUT_OFFSET
#endif
#ifdef FUSED_OP_1_OUTPUT_SIZES_DATA
#undef FUSED_OP_1_OUTPUT_SIZES_DATA
#endif
#ifdef FUSED_OP_1_OUTPUT_PITCHES
#undef FUSED_OP_1_OUTPUT_PITCHES
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_BEFORE
#undef FUSED_OP_1_OUTPUT_PAD_BEFORE
#endif
#ifdef FUSED_OP_1_OUTPUT_PAD_AFTER
#undef FUSED_OP_1_OUTPUT_PAD_AFTER
#endif
#ifdef FUSED_OP1_DECLS
#undef FUSED_OP1_DECLS
#endif
#ifdef FUSED_OP1_ARGS
#undef FUSED_OP1_ARGS
#endif
#ifdef FUSED_OPS_DECLS
#undef FUSED_OPS_DECLS
#endif
#ifdef FUSED_OPS_ARGS
#undef FUSED_OPS_ARGS
#endif
#ifdef HAS_FUSED_OPS
#undef HAS_FUSED_OPS
#endif
#ifdef HAS_FUSED_OPS_DECLS
#undef HAS_FUSED_OPS_DECLS
#endif
#ifdef X_BLOCK_SIZE
#undef X_BLOCK_SIZE
#endif
#ifdef SLM_DIV_FACTOR
#undef SLM_DIV_FACTOR
#endif
#ifdef SUB_GROUP_SIZE
#undef SUB_GROUP_SIZE
#endif
#ifdef WORK_GROUP_SIZE
#undef WORK_GROUP_SIZE
#endif
#ifdef PADDED_INPUT
#undef PADDED_INPUT
#endif
#ifdef PADDED_OUTPUT
#undef PADDED_OUTPUT
#endif
#ifdef NON_UNIT_FUSED_OP_SPATIAL
#undef NON_UNIT_FUSED_OP_SPATIAL
#endif
#ifdef IC_BLOCKS
#undef IC_BLOCKS
#endif
#ifdef INPUT_LEFTOVERS
#undef INPUT_LEFTOVERS
#endif
 