pub mod utils;
mod tests;

use crate::utils::
    {setup_cpu, compose_imm, 
    compose_rr, compose_b};

fn main() {
    let mut cpu = setup_cpu();
    // addi x10, x10, 10
    let inst = compose_imm(19, 0, 10, 10, 10);
    cpu.memory.store(cpu.pc, inst, 4);
    // addi x11, x0, 0
    let inst = compose_imm(19, 0, 11, 0, 0);
    cpu.memory.store(cpu.pc + 4, inst, 4);
    // addi x12, x0, 1
    let inst = compose_imm(19, 0, 12, 0, 1);
    cpu.memory.store(cpu.pc + 8, inst, 4);
    // add x13, x12, x11
    let inst = compose_rr(51, 0, 0, 13, 12, 11);
    cpu.memory.store(cpu.pc + 12, inst, 4);
    // add x14, x0, x12
    let inst = compose_rr(51, 0, 0, 14, 0, 12);
    cpu.memory.store(cpu.pc + 16, inst, 4);
    // add x12, x0, x13
    let inst = compose_rr(51, 0, 0, 12, 0, 13);
    cpu.memory.store(cpu.pc + 20, inst, 4);
    // add x11, x0, x14
    let inst = compose_rr(51, 0, 0, 11, 0, 14);
    cpu.memory.store(cpu.pc + 24, inst, 4);
    // addi x10, x10, -1
    let inst = compose_imm(19, 0, 10, 10, -1);
    cpu.memory.store(cpu.pc + 28, inst, 4);
    // bge x10, x0, -10
    let offset = -10;
    let inst = compose_b(99, 5, 10, 0, offset as u32);
    cpu.memory.store(cpu.pc + 32, inst, 4);

    cpu.run_debug(4);

    println!("result of fibonacci(10): {}", cpu.registers[13]);
}
