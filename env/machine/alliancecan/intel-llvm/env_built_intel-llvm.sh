module purge
module load StdEnv/2023
module load intel/2025.2.0
module load cuda/12.9

echo ${EBROOTINTELMINCOMPILERS}
echo ${EBROOTINTELMINCOMPILERS}/compiler/2025.2/bin/icpx
echo ${EBROOTINTELMINCOMPILERS}/compiler/2025.2/bin/icx

function shamconfigure {
	cmake \
	-S $SHAMROCK_DIR \
	-B $BUILD_DIR \
	-G "Unix Makefiles" \
	-DSHAMROCK_ENABLE_BACKEND=SYCL \
	-DSYCL_IMPLEMENTATION=IntelLLVM \
	-DINTEL_LLVM_PATH=${EBROOTINTELMINCOMPILERS}\
	-DCMAKE_CXX_COMPILER=${EBROOTINTELMINCOMPILERS}/compiler/2025.2/bin/icpx \
	-DCMAKE_C_COMPILER=${EBROOTINTELMINCOMPILERS}/compiler/2025.2/bin/icx \
	-DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=nvidia_gpu_sm_90" \
	# -DCMAKE_EXE_LINKER_FLAGS="-Wl,--copy-dt-needed-entries" \   # Only works on nibi cluster
	-DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
	-DBUILD_TEST=Yes \
	"${CMAKE_OPT[@]}"
}

function shammake {
	(cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}")
}
