//! The seccomp filter bwrap loads for every sandboxed command: no `AF_UNIX`
//! sockets, so no host Unix socket bound in is reachable, and no `io_uring`.

use std::fs::File;
use std::io::{self, Seek, Write};
use std::os::fd::OwnedFd;

use rustix::fs::MemfdFlags;

#[cfg(target_arch = "x86_64")]
const AUDIT_ARCH_NATIVE: u32 = 0xc000_003e;
#[cfg(target_arch = "aarch64")]
const AUDIT_ARCH_NATIVE: u32 = 0xc000_00b7;

#[cfg(target_arch = "x86_64")]
const SYS_SOCKET: u32 = 41;
#[cfg(target_arch = "aarch64")]
const SYS_SOCKET: u32 = 198;

const SYS_IO_URING_SETUP: u32 = 425;

/// Set in the number of every x32 system call on `x86_64`.
const X32_SYSCALL_BIT: u32 = 0x4000_0000;

const AF_UNIX: u32 = 1;
const EACCES: u32 = 13;
const ENOSYS: u32 = 38;

/// Offsets into `struct seccomp_data`; `args[0]`'s low half on a
/// little-endian machine, which is all of `socket`'s `int domain`.
const NR_OFFSET: u32 = 0;
const ARCH_OFFSET: u32 = 4;
const ARG0_OFFSET: u32 = 16;

const SECCOMP_RET_KILL_PROCESS: u32 = 0x8000_0000;
const SECCOMP_RET_ERRNO: u32 = 0x0005_0000;
const SECCOMP_RET_ALLOW: u32 = 0x7fff_0000;

const BPF_LD_W_ABS: u16 = 0x20;
const BPF_JMP_JEQ_K: u16 = 0x15;
const BPF_JMP_JSET_K: u16 = 0x45;
const BPF_RET_K: u16 = 0x06;

/// Kills any process on a foreign ABI, whose arguments the filter could
/// misread, refuses `socket(AF_UNIX, ...)` with `EACCES` and
/// `io_uring_setup` with `ENOSYS`, and allows everything else.
const FILTER: [Instruction; 13] = [
    Instruction::load(ARCH_OFFSET),
    Instruction::jump_if_equal(AUDIT_ARCH_NATIVE, 1, 0),
    Instruction::ret(SECCOMP_RET_KILL_PROCESS),
    Instruction::load(NR_OFFSET),
    Instruction::jump_if_any_set(X32_SYSCALL_BIT, 0, 1),
    Instruction::ret(SECCOMP_RET_KILL_PROCESS),
    Instruction::jump_if_equal(SYS_IO_URING_SETUP, 0, 1),
    Instruction::ret(SECCOMP_RET_ERRNO | ENOSYS),
    Instruction::jump_if_equal(SYS_SOCKET, 0, 3),
    Instruction::load(ARG0_OFFSET),
    Instruction::jump_if_equal(AF_UNIX, 0, 1),
    Instruction::ret(SECCOMP_RET_ERRNO | EACCES),
    Instruction::ret(SECCOMP_RET_ALLOW),
];

/// One classic BPF `struct sock_filter`; jump offsets count instructions
/// after this one.
#[derive(Debug, Clone, Copy)]
struct Instruction {
    code: u16,
    jump_true: u8,
    jump_false: u8,
    operand: u32,
}

impl Instruction {
    const fn load(offset: u32) -> Self {
        Self::new(BPF_LD_W_ABS, 0, 0, offset)
    }

    const fn jump_if_equal(value: u32, jump_true: u8, jump_false: u8) -> Self {
        Self::new(BPF_JMP_JEQ_K, jump_true, jump_false, value)
    }

    const fn jump_if_any_set(mask: u32, jump_true: u8, jump_false: u8) -> Self {
        Self::new(BPF_JMP_JSET_K, jump_true, jump_false, mask)
    }

    const fn ret(action: u32) -> Self {
        Self::new(BPF_RET_K, 0, 0, action)
    }

    const fn new(code: u16, jump_true: u8, jump_false: u8, operand: u32) -> Self {
        Self {
            code,
            jump_true,
            jump_false,
            operand,
        }
    }

    fn to_ne_bytes(self) -> [u8; 8] {
        let [code_0, code_1] = self.code.to_ne_bytes();
        let [op_0, op_1, op_2, op_3] = self.operand.to_ne_bytes();
        [
            code_0,
            code_1,
            self.jump_true,
            self.jump_false,
            op_0,
            op_1,
            op_2,
            op_3,
        ]
    }
}

/// A close-on-exec memfd holding [`FILTER`], positioned at its start, for
/// bwrap's `--seccomp`, which reads it once.
pub(super) fn filter_fd() -> io::Result<OwnedFd> {
    let mut file = File::from(rustix::fs::memfd_create(
        "assistd-seccomp",
        MemfdFlags::CLOEXEC,
    )?);
    let program: Vec<u8> = FILTER
        .iter()
        .flat_map(|instruction| instruction.to_ne_bytes())
        .collect();
    file.write_all(&program)?;
    file.rewind()?;
    Ok(file.into())
}
