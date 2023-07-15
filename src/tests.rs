#[cfg(test)]
mod tests {
    use crate::utils::
        {setup_cpu, compose_imm, 
        compose_imm_f7, compose_rr, compose_b, 
        compose_j, compose_u, compose_s};

    #[test]
    fn test_add() {
        let mut cpu = setup_cpu();
        // addi x1, x0, 1
        let inst = compose_imm(19, 0, 1, 0, 1);
        cpu.decode_execute(inst);

        // addi x12, x1, 9
        let inst = compose_imm(19, 0, 12, 1, 9);
        cpu.decode_execute(inst);

        // add x1, x12, x1
        let inst = compose_rr(51, 0, 0, 1, 12, 1);
        cpu.decode_execute(inst);
        assert_eq!(11, cpu.registers[1]);

        // add x1, x12, x1
        let inst = compose_rr(51, 0, 0, 1, 12, 1);
        cpu.decode_execute(inst);
        assert_eq!(21, cpu.registers[1]);

        // addi x12, x0, -21
        let inst = compose_imm(19, 0, 12, 0, -21);
        cpu.decode_execute(inst);

        // add x1, x12, x1
        let inst = compose_rr(51, 0, 0, 1, 12, 1);
        cpu.decode_execute(inst);
        assert_eq!(0, cpu.registers[1]);

        // addi x1, x0, 2047
        let inst = compose_imm(19, 0, 1, 0, 2047);
        cpu.decode_execute(inst);
        assert_eq!(2047, cpu.registers[1]);

        // slli x1, x1, 20
        let inst = compose_imm(19, 1, 1, 1, 20);
        cpu.decode_execute(inst);
        assert_eq!(0x7ff00000, cpu.registers[1]);

        // overflow test:
        // add x1, x1, x1
        let inst = compose_rr(51, 0, 0, 1, 1, 1);
        cpu.decode_execute(inst);
    }

    #[test]
    fn test_sub() {
        let mut cpu = setup_cpu();
        // addi x1, x0, 1 (x1 = 1)
        let inst = compose_imm(19, 0, 1, 0, 1);
        cpu.decode_execute(inst);
        // addi x12, x1, 9 (x12 = 10)
        let inst = compose_imm(19, 0, 12, 1, 9);
        cpu.decode_execute(inst);
        
        // sub x1, x1, x12 (x1 = -9)
        let inst = compose_rr(51, 0, 32, 1, 1, 12);
        cpu.decode_execute(inst);
        assert_eq!(-9, cpu.registers[1]);

        // addi x13, x0, -13 (x13 = -13)
        let inst = compose_imm(19, 0, 13, 0, -13);
        cpu.decode_execute(inst);
        assert_eq!(-13, cpu.registers[13]);

        // sub x1, x1, x13 (x1 = 4)
        let inst = compose_rr(51, 0, 32, 1, 1, 13);
        cpu.decode_execute(inst);
        assert_eq!(4, cpu.registers[1]);
    }

    #[test]
    fn test_sll() {
        let mut cpu = setup_cpu();
        // sll x1, x0, x1
        let inst = compose_rr(51, 1, 0, 1, 0, 1);
        cpu.decode_execute(inst);
        assert_eq!(0, cpu.registers[1]);

        // addi x12, x0, 9
        let inst = compose_imm(19, 0, 12, 0, 9);
        cpu.decode_execute(inst);
        assert_eq!(9, cpu.registers[12]);

        // addi x1, x0, 4
        let inst = compose_imm(19, 0, 1, 0, 4);
        cpu.decode_execute(inst);
        assert_eq!(4, cpu.registers[1]);

        // sll x12, x12, x1
        let inst = compose_rr(51, 1, 0, 12, 12, 1);
        cpu.decode_execute(inst);
        assert_eq!(144, cpu.registers[12]);

        // sll x12, x12, x12
        // should shift 144 left by 16
        let inst = compose_rr(51, 1, 0, 12, 12, 12);
        cpu.decode_execute(inst);
        assert_eq!(0x00900000, cpu.registers[12]);
    }

    #[test]
    fn test_slt() {
        let mut cpu = setup_cpu();

        // slt x1, x0, x0
        let inst = compose_rr(51, 2, 0, 1, 0, 0);
        cpu.decode_execute(inst);
        assert_eq!(0, cpu.registers[1]);

        // addi x5, x5, -15
        let inst = compose_imm(19, 0, 5, 5, -15);
        cpu.decode_execute(inst);
        assert_eq!(-15, cpu.registers[5]);

        // addi x12, x12, -20
        let inst = compose_imm(19, 0, 12, 12, -20);
        cpu.decode_execute(inst);
        assert_eq!(-20, cpu.registers[12]);

        // slt x1, x12, x5
        let inst = compose_rr(51, 2, 0, 1, 12, 5);
        cpu.decode_execute(inst);
        assert_eq!(1, cpu.registers[1]);
    }
    #[test]
    fn test_sltu() {
        let mut cpu = setup_cpu();

        // sltu x1, x0, x1
        let inst = compose_rr(51, 3, 0, 1, 0, 1);
        cpu.decode_execute(inst);
        assert_eq!(0, cpu.registers[1]);

        // addi x1, x1, -15
        let inst = compose_imm(19, 0, 1, 1, -15);
        cpu.decode_execute(inst);
        assert_eq!(-15, cpu.registers[1]);

        // sltu x1, x0, x1
        let inst = compose_rr(51, 3, 0, 1, 0, 1);
        cpu.decode_execute(inst);
        assert_eq!(1, cpu.registers[1]);
    }

    #[test]
    fn test_xor() {
        let mut cpu = setup_cpu();

        // addi x1, x0, 3
        let inst = compose_imm(19, 0, 1, 0, 3);
        cpu.decode_execute(inst);
        assert_eq!(3, cpu.registers[1]);

        // addi x5, x0, -1
        let inst = compose_imm(19, 0, 5, 0, -1);
        cpu.decode_execute(inst);
        assert_eq!(-1, cpu.registers[5]);

        // xor x1, x1, x5
        let inst = compose_rr(51, 4, 0, 1, 1, 5);
        cpu.decode_execute(inst);
        assert_eq!(0xfffffffc, cpu.registers[1] as u32);

        // addi x1, x0, 3
        let inst = compose_imm(19, 0, 1, 0, 3);
        cpu.decode_execute(inst);
        assert_eq!(3, cpu.registers[1]);
    }

    #[test]
    fn test_srl() {
        let mut cpu = setup_cpu();
        // addi x1, x0, 3 (x1 = 3)
        let inst = compose_imm(19, 0, 1, 0, 3);
        cpu.decode_execute(inst);
        assert_eq!(3, cpu.registers[1]);

        // addi x5, x0, 1 (x5 = 1)
        let inst = compose_imm(19, 0, 5, 0, 1);
        cpu.decode_execute(inst);
        assert_eq!(1, cpu.registers[5]);

        // srl x1, x1, x5 ( x1 = x1 >> x5)
        let inst = compose_rr(51, 5, 0, 1, 1, 5);
        cpu.decode_execute(inst);
        assert_eq!(1, cpu.registers[1]);

        // addi x1, x0, 64 (x1 = 64)
        let inst = compose_imm(19, 0, 1, 0, 64);
        cpu.decode_execute(inst);
        assert_eq!(64, cpu.registers[1]);

        // srl x5, x5, x1 (x1 = x5 >> x1)
        let inst = compose_rr(51, 5, 0, 5, 5, 1);
        cpu.decode_execute(inst);
        assert_eq!(1, cpu.registers[5]);
    }

    #[test]
    fn test_srla() {
        let mut cpu = setup_cpu();
        // addi x1, x0, -1 (x1 = -1)
        let inst = compose_imm(19, 0, 1, 0, -1);
        cpu.decode_execute(inst);
        assert_eq!(-1, cpu.registers[1]);

        // addi x5, x0, 1 (x5 = 1)
        let inst = compose_imm(19, 0, 5, 0, 1);
        cpu.decode_execute(inst);
        assert_eq!(1, cpu.registers[5]);

        // srla x1, x1, x5 ( x1 = x1 >> x5)
        let inst = compose_rr(51, 5, 32, 1, 1, 5);
        cpu.decode_execute(inst);
        // 0xffffffff >> 1 = 0xffffffff
        // 0x7fffffff
        assert_eq!(-1, cpu.registers[1]);

    }

    #[test]
    fn test_or() {
        let mut cpu = setup_cpu();

        // addi x12, x12, -1
        let inst = compose_imm(19, 0, 12, 12, -1);
        cpu.decode_execute(inst);
        assert_eq!(-1, cpu.registers[12]);

        // or x1, x0, x12
        let inst = compose_rr(51, 6, 0, 1, 0, 12);
        cpu.decode_execute(inst);
        assert_eq!(-1, cpu.registers[1]);

    }

    #[test]
    fn test_and() {
        let mut cpu = setup_cpu();

        // addi x12, x12, -1
        let inst = compose_imm(19, 0, 12, 12, -1);
        cpu.decode_execute(inst);
        assert_eq!(-1, cpu.registers[12]);

        // and x1, x12, x0
        let inst = compose_rr(51, 7, 0, 1, 12, 0);
        cpu.decode_execute(inst);
        assert_eq!(0, cpu.registers[1]);

    }



    #[test]
    fn test_addi() {
        let mut cpu = setup_cpu();
        // addi x1, x0, 1
        let inst = compose_imm(19, 0, 1, 0, 1);
        cpu.decode_execute(inst);
        assert_eq!(cpu.registers[1], 1);

        // addi x12, x1, 9
        let inst = compose_imm(19, 0, 12, 1, 9);
        cpu.decode_execute(inst);
        assert_eq!(cpu.registers[12], 10);
        assert_eq!(cpu.registers[1], 1);

        // addi x13, 10, -9
        let inst = compose_imm(19, 0, 13, 1, -9);
        cpu.decode_execute(inst);
        assert_eq!(cpu.registers[13], -8);
        assert_eq!(cpu.registers[1], 1);

        // addi x2, x0, -1
        let inst = compose_imm(19, 0, 2, 0, -1);
        cpu.decode_execute(inst);
        assert_eq!(-1, cpu.registers[2]);

        // srli x2, x2, 1
        let inst = compose_imm_f7(19, 5, 0, 2, 2, 1);
        cpu.decode_execute(inst);
        assert_eq!(0x7fffffff, cpu.registers[2]);

        // addi x2, x2, 1
        let inst = compose_imm(19, 0, 2, 2, 1);
        cpu.decode_execute(inst);
        assert_eq!(0x80000000 as u32, cpu.registers[2] as u32);
    }

    #[test]
    fn test_slli() {
        let mut cpu = setup_cpu();
        // slli x1, x0, 1
        let inst = compose_imm(19, 1, 1, 0, 1);
        cpu.decode_execute(inst);
        assert_eq!(0, cpu.registers[1]);

        // addi x12, x0, 9
        let inst = compose_imm(19, 0, 12, 0, 9);
        cpu.decode_execute(inst);
        assert_eq!(9, cpu.registers[12]);

        // slli x12, x12, 2
        let inst = compose_imm(19, 1, 12, 12, 2);
        cpu.decode_execute(inst);
        assert_eq!(36, cpu.registers[12]);
    }

    #[test]
    fn test_slti() {
        let mut cpu = setup_cpu();
        // slti x1, x0, -1
        let inst = compose_imm(19, 2, 1, 0, -1);
        cpu.decode_execute(inst);
        assert_eq!(0, cpu.registers[1]);

        // slti x1, x0, 3
        let inst = compose_imm(19, 2, 1, 0, 3);
        cpu.decode_execute(inst);
        assert_eq!(1, cpu.registers[1]);
    }

    #[test]
    fn test_sltiu() {
        let mut cpu = setup_cpu();

        // addi x12, x0, 2047
        let inst = compose_imm(19, 0, 12, 0, 2047);
        cpu.decode_execute(inst);
        assert_eq!(2047, cpu.registers[12] as u32);

        // addi x12, x12, 2047
        let inst = compose_imm(19, 0, 12, 12, 2047);
        cpu.decode_execute(inst);
        assert_eq!(4094, cpu.registers[12] as u32);

        // sltiu x12, x12, 4095 
        let inst = compose_imm(19, 3, 12, 12, 4095);
        cpu.decode_execute(inst);
        assert_eq!(1, cpu.registers[12]);
    }
    #[test]
    fn test_xori() {
        let mut cpu = setup_cpu();
        // xori x12, x12, -1
        let inst = compose_imm(19, 4, 12, 12, -1);
        cpu.decode_execute(inst);
        assert_eq!(0xffffffff, cpu.registers[12] as u32);
    }
    #[test]
    fn test_srli() {
        let mut cpu = setup_cpu();
        // addi x12, x0, 2047
        let inst = compose_imm(19, 0, 12, 0, 2047);
        cpu.decode_execute(inst);
        assert_eq!(2047, cpu.registers[12]);

        // srli x12, x12, 1
        let inst = compose_imm_f7(19, 5, 0, 12, 12, 1);
        cpu.decode_execute(inst);
        assert_eq!(1023, cpu.registers[12]);

        // addi x12, x0, -2048
        let inst = compose_imm(19, 0, 12, 0, -2048);
        cpu.decode_execute(inst);
        assert_eq!(-2048, cpu.registers[12]);

        // srli x12, x12, 1
        let inst = compose_imm_f7(19, 5, 0, 12, 12, 1);
        cpu.decode_execute(inst);
        assert!(cpu.registers[12] > 0);
    }
    #[test]
    fn test_srlai() {
        let mut cpu = setup_cpu();
        // addi x12, x0, 2047
        let inst = compose_imm(19, 0, 12, 0, 2047);
        cpu.decode_execute(inst);
        assert_eq!(2047, cpu.registers[12]);

        // srlai x12, x12, 1
        let inst = compose_imm_f7(19, 5, 32, 12, 12, 1);
        cpu.decode_execute(inst);
        assert_eq!(1023, cpu.registers[12]);

        // addi x12, x0, -2048
        let inst = compose_imm(19, 0, 12, 0, -2048);
        cpu.decode_execute(inst);
        assert_eq!(-2048, cpu.registers[12]);

        // srlai x12, x12, 1
        let inst = compose_imm_f7(19, 5, 32, 12, 12, 1);
        cpu.decode_execute(inst);
        assert!(cpu.registers[12] < 0);
    }
    #[test]
    fn test_ori() {
        let mut cpu = setup_cpu();
        // ori x12, x12, -1
        let inst = compose_imm(19, 6, 12, 12, -1);
        cpu.decode_execute(inst);
        assert_eq!(0xffffffff, cpu.registers[12] as u32);
    }
    #[test]
    fn test_andi() {
        let mut cpu = setup_cpu();
        // addi x12, x0, 2047
        let inst = compose_imm(19, 0, 12, 0, 2047);
        cpu.decode_execute(inst);
        assert_eq!(2047, cpu.registers[12]);

        // andi x12, x12, 0
        let inst = compose_imm(19, 7, 12, 12, 0);
        cpu.decode_execute(inst);
        assert_eq!(0, cpu.registers[12] as u32);
    }

    #[test]
    fn test_loads_stores() {
        let mut cpu = setup_cpu();

        // addi x12, x0, 2047
        let inst = compose_imm(19, 0, 12, 0, 2047);
        cpu.decode_execute(inst);
        assert_eq!(2047, cpu.registers[12]);

        // sw x12, 0(sp)
        let inst = compose_s(35, 2, 12, 0, 2);
        cpu.decode_execute(inst);

        // lw x11, 0(sp)
        let inst = compose_imm(3, 2, 11, 2, 0);
        cpu.decode_execute(inst);
        assert_eq!(2047, cpu.registers[11]);

        // lbu x11, 0(sp)
        let inst = compose_imm(3, 4, 11, 2, 0);
        cpu.decode_execute(inst);
        assert_eq!(0x000000ff, cpu.registers[11]);

        // lb x11, 0(sp)
        let inst = compose_imm(3, 0, 11, 2, 0);
        cpu.decode_execute(inst);
        assert_eq!(-1, cpu.registers[11]);

        // lhu x11, 0(sp)
        let inst = compose_imm(3, 5, 11, 2, 0);
        cpu.decode_execute(inst);
        assert_eq!(0x000007ff, cpu.registers[11]);

        // lh x11, 0(sp)
        let inst = compose_imm(3, 1, 11, 2, 0);
        cpu.decode_execute(inst);
        assert_eq!(0x000007ff, cpu.registers[11]);
    }

    #[test]
    fn test_auipc() {
        let mut cpu = setup_cpu();
        // auipc x5, 1<<19
        let inst = compose_u(23, 5, 1 << 19);
        cpu.decode_execute(inst);
        assert_eq!(cpu.pc + (1 << 31), cpu.registers[5] as u32);
    }

    #[test]
    fn test_lui() {
        let mut cpu = setup_cpu();
        // lui x5, 1<<19
        let inst = compose_u(55, 5, 1 << 19);
        cpu.decode_execute(inst);
        assert_eq!((1 << 31), cpu.registers[5] as u32);
    }

    #[test]
    fn test_jumps() {
        let mut cpu = setup_cpu();

        // jal x1, -4
        let initial_pc = cpu.pc;
        let offset = -4;
        let inst = compose_j(111, 1, offset as u32);
        cpu.decode_execute(inst);
        assert_eq!(initial_pc - 12, cpu.pc);

        // lui x12, 0x00400
        let inst = compose_u(55, 12, 0x00000400);
        cpu.decode_execute(inst);
        assert_eq!(initial_pc, cpu.registers[12] as u32);

        // jalr x0, 0(x12)
        let inst = compose_imm(103, 0, 0, 12, 0);
        cpu.decode_execute(inst);
        assert_eq!(initial_pc - 4, cpu.pc);
    }

    #[test]
    fn test_branches() {
        let mut cpu = setup_cpu();
        // addi x10, x0, 2047
        let inst = compose_imm(19, 0, 10, 0, 2047);
        cpu.decode_execute(inst);
        assert_eq!(2047, cpu.registers[10]);

        // addi x11, x0, 2043
        let inst = compose_imm(19, 0, 11, 0, 2043);
        cpu.decode_execute(inst);
        assert_eq!(2043, cpu.registers[11]);

        // addi x12, x0, 2047
        let inst = compose_imm(19, 0, 12, 0, 2047);
        cpu.decode_execute(inst);
        assert_eq!(2047, cpu.registers[12]);

        // beq x10, x11, -2
        let offset = -2;
        let inst = compose_b(99, 0, 10, 11, offset as u32);
        let initial_pc = cpu.pc;
        cpu.decode_execute(inst);
        assert_eq!(initial_pc, cpu.pc);

        // beq x10, x12, -2
        let offset = -2;
        let inst = compose_b(99, 0, 10, 12, offset as u32);
        let initial_pc = cpu.pc;
        cpu.decode_execute(inst);
        assert_eq!(initial_pc-8, cpu.pc);

        // bneq x10, x11, -2048
        let offset = -2048;
        let inst = compose_b(99, 1, 10, 11, offset as u32);
        let initial_pc = cpu.pc;
        cpu.decode_execute(inst);
        assert_eq!(initial_pc-(4096)-4, cpu.pc);

        // blt x10, x12, -2
        let offset = -2;
        let inst = compose_b(99, 4, 10, 12, offset as u32);
        let initial_pc = cpu.pc;
        cpu.decode_execute(inst);
        assert_eq!(initial_pc, cpu.pc);

        // bge x10, x12, -2
        let offset = -2;
        let inst = compose_b(99, 5, 10, 12, offset as u32);
        let initial_pc = cpu.pc;
        cpu.decode_execute(inst);
        assert_eq!(initial_pc-8, cpu.pc);

        // addi x13, x13, -1
        let inst = compose_imm(19, 0, 13, 13, -1);
        cpu.decode_execute(inst);
        assert_eq!(-1, cpu.registers[13]);

        // bltu x13, x0, -2
        let offset = -2;
        let inst = compose_b(99, 6, 13, 0, offset as u32);
        let initial_pc = cpu.pc;
        cpu.decode_execute(inst);
        assert_eq!(initial_pc, cpu.pc);

        // bgeu x13, x0, -2
        let offset = -2;
        let inst = compose_b(99, 7, 13, 0, offset as u32);
        let initial_pc = cpu.pc;
        cpu.decode_execute(inst);
        assert_eq!(initial_pc-8, cpu.pc);
    }
}
