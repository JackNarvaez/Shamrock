# Everything before this line will be provided by the new-env script

module purge
module load StdEnv/2023 llvm/20.1.8 || return
module load nvhpc/25.1 openmpi/5.0.3 cuda/12.6 || return

export ACPP_VERSION=v25.10.0
export ACPP_GIT_DIR=$BUILD_DIR/.env/acpp-git
export ACPP_BUILD_DIR=$BUILD_DIR/.env/acpp-builddir
export ACPP_INSTALL_DIR=$BUILD_DIR/.env/acpp-installdir

export PATH=$ACPP_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$ACPP_INSTALL_DIR/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$ACPP_INSTALL_DIR:$CMAKE_PREFIX_PATH

export SHAM_MPI_INIT_STRATEGY=mpifirst

function setupcompiler {
    clone_acpp || return
    cmake -S ${ACPP_GIT_DIR} -B ${ACPP_BUILD_DIR} \
        -DCMAKE_INSTALL_PREFIX=${ACPP_INSTALL_DIR} \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DACPP_LLD_PATH=/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/ld.lld \
        -DWITH_CUDA_BACKEND=ON || return
    (cd ${ACPP_BUILD_DIR} && $MAKE_EXEC "${MAKE_OPT[@]}" && $MAKE_EXEC install) || return
}

if [ ! -f "$ACPP_INSTALL_DIR/bin/acpp" ]; then
    echo " ----- acpp is not configured, compiling it ... -----"
    setupcompiler || return
    echo " ----- acpp configured ! -----"
fi

function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=ACPPDirect \
        -DCMAKE_CXX_COMPILER="${ACPP_INSTALL_DIR}/bin/acpp" \
        -DCMAKE_CXX_FLAGS="${SHAMROCK_CXX_FLAGS}" \
        -DACPP_PATH="${ACPP_INSTALL_DIR}" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DCXX_FLAG_ARCH_NATIVE=off \
        -DBUILD_TEST=Yes \
        "${CMAKE_OPT[@]}" || return
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}") || return
}
