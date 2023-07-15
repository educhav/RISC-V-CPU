use risc_v_cpu::*;

pub fn setup_cpu() -> CPU {
    let mut cpu = CPU {
        registers: vec![0; 32],
        pc: 0,
        memory: Box::new(RAM {
            memory: vec![0; 2 << 31]
        })
    };
    cpu.initialize(0x7ffffff0, 0x10000000, 0x00400000);
    return cpu;
}
pub fn disassemble(inst: u32) -> String {
    let opcode = inst & OPCODE_MASK;
    match opcode {
        // Load instructions: lb, lh, lw, lbu, lhu
        3 => {
            let rd = (inst & RD_MASK) >> 7;
            let funct3 = (inst & FUNCT3_MASK) >> 12;
            let rs1 = (inst & RS1_MASK) >> 15;
            let imm = ((inst & IMM12_MASK) as i32) >> 20;
            match funct3 {
                0 => format!("lb x{}, {}(x{})", rd, imm, rs1),
                1 => format!("lh x{}, {}(x{})", rd, imm, rs1),
                2 => format!("lw x{}, {}(x{})", rd, imm, rs1),
                4 => format!("lbu x{}, {}(x{})", rd, imm, rs1),
                5 => format!("lhu x{}, {}(x{})", rd, imm, rs1),
                _ => format!("Unrecognized funct3")
            }
        }
        // Store instructions: sb, sh, sw
        35 => {
            let mut imm = 0;
            let imm4_0 = (inst & IMM12_4_0_MASK) >> 7;
            imm |= imm4_0;
            let imm11_5 = (inst & IMM12_11_5_MASK) >> 20;
            imm |= imm11_5;
            let rs1 = (inst & RS1_MASK) >> 15;
            let rs2 = (inst & RS2_MASK) >> 20;
            let funct3 = (inst & FUNCT3_MASK) >> 12;
            match funct3 {
                0 => format!("sb x{}, ({})x{}", rs1, CPU::sext(imm, 0xfffff800), rs2),
                1 => format!("sh x{}, ({})x{}", rs1, CPU::sext(imm, 0xfffff800), rs2),
                2 => format!("sw x{}, ({})x{}", rs1, CPU::sext(imm, 0xfffff800), rs2),
                _ => format!("Unrecognized funct3")
            }
        }
        // Immediate ops: addi, slli, slti, sltiu, xori, srli, srai, ori, andi
        19 => {
            let rd = (inst & RD_MASK) >> 7;
            let funct3 = (inst & FUNCT3_MASK) >> 12;
            let rs1 = (inst & RS1_MASK) >> 15;
            let imm12 = (inst & IMM12_MASK) >> 20;
            let imm5 = (inst & IMM5_MASK) >> 20;
            match funct3 {
                0 => format!("addi x{}, x{}, {}", rd, rs1, CPU::sext(imm12, 0xfffff800)),
                1 => format!("slli x{}, x{}, {}", rd, rs1, imm5),
                2 => format!("slti x{}, x{}, {}", rd, rs1, CPU::sext(imm12, 0xfffff800)),
                3 => format!("sltiu x{}, x{}, {}", rd, rs1, imm12),
                4 => format!("xori x{}, x{}, {}", rd, rs1, CPU::sext(imm12, 0xfffff800)),
                5 => {
                    let funct7 = (inst & FUNCT7_MASK) >> 25;
                    match funct7 {
                        0 => format!("srli x{}, x{}, {}", rd, rs1, imm5),
                        32 => format!("srlai x{}, x{}, {}", rd, rs1, imm5),
                        _ => panic!("CPU exception: unrecognized funct7")
                    }
                }
                6 => format!("ori x{}, x{}, {}", rd, rs1, CPU::sext(imm12, 0xfffff800)),
                7 => format!("andi x{}, x{}, {}", rd, rs1, CPU::sext(imm12, 0xfffff800)),
                _ => format!("CPU exception: unrecognized funct3")
            }
        }
        // RR ops: add, sub, sll, slt, sltu, xor, srl, sra, or, and
        51 => {
            let rd = (inst & RD_MASK) >> 7;
            let funct3 = (inst & FUNCT3_MASK) >> 12;
            let rs1 = (inst & RS1_MASK) >> 15;
            let rs2 = (inst & RS2_MASK) >> 20;
            let funct7 = (inst & FUNCT7_MASK) >> 25;
            match funct3 {
                0 => {
                    match funct7 {
                        0 => format!("add x{}, x{}, x{}", rd, rs1, rs2),
                        32 => format!("sub x{}, x{}, x{}", rd, rs1, rs2),
                        _ => panic!("CPU exception: unrecognized funct7")
                    }
                }
                1 => format!("sll x{}, x{}, x{}", rd, rs1, rs2),
                2 => format!("slt x{}, x{}, x{}", rd, rs1, rs2),
                3 => format!("sltu x{}, x{}, x{}", rd, rs1, rs2),
                4 => format!("xor x{}, x{}, x{}", rd, rs1, rs2),
                5 => {
                    match funct7 {
                        0 => format!("srl x{}, x{}, x{}", rd, rs1, rs2),
                        32 => format!("srla x{}, x{}, x{}", rd, rs1, rs2),
                        _ => format!("Unrecognized funct7")
                    }
                }
                6 => format!("or x{}, x{}, x{}", rd, rs1, rs2),
                7 => format!("and x{}, x{}, x{}", rd, rs1, rs2),
                _ => format!("Unrecognized funct3")
                // _ => panic!("CPU exception: unrecognized funct3")
            }
        }
        // auipc 
        23 => {
            let rd = (inst & RD_MASK) >> 7;
            let imm = inst & IMM20_MASK;
            format!("auipc x{}, {:x}", rd, imm)
        }
        // lui
        55 => {
            let rd = (inst & RD_MASK) >> 7;
            let imm = inst & IMM20_MASK;
            format!("lui x{}, {:x}", rd, imm)
        }
        // branches: beq, bne, blt, bge, bltu, bgeu
        99 => {
            let mut imm = 0;
            imm |= (inst & SB_IMM20_4_1_MASK) >> 8;
            imm |= (inst & SB_IMM20_10_5_MASK) >> 21;
            imm |= (inst & SB_IMM20_11_MASK) << 3;
            imm |= (inst & SB_IMM20_12_MASK) >> 20;
            let imm = CPU::sext(imm << 1, 0xfffff000);
            let funct3 = (inst & FUNCT3_MASK) >> 12;
            let rs1 = (inst & RS1_MASK) >> 15;
            let rs2 = (inst & RS2_MASK) >> 20;
            match funct3 {
                0 => format!("beq x{}, x{}, {}", rs1, rs2, imm),
                1 => format!("bneq x{}, x{}, {}", rs1, rs2, imm),
                4 => format!("blt x{}, x{}, {}", rs1, rs2, imm),
                5 => format!("bge x{}, x{}, {}", rs1, rs2, imm),
                6 => format!("bltu x{}, x{}, {}", rs1, rs2, imm),
                7 => format!("bgeu x{}, x{}, {}", rs1, rs2, imm),
                _ => format!("CPU exception: unrecognized funct3")
            }
        }
        // jalr
        103 => {
            let rd = (inst & RD_MASK) >> 7;
            let rs1 = (inst & RS1_MASK) >> 15;
            let imm = ((inst & IMM12_MASK) as i32) >> 20;
            format!("jalr x{}, ({})x{}", rd, imm, rs1)
        }
        // jal
        111 => {
            let rd = (inst & RD_MASK) >> 7;
            let mut imm = 0;
            imm |= (inst & IMM20_10_1_MASK) >> 21;
            imm |= (inst & IMM20_11_MASK) >> 10;
            imm |= (inst & IMM20_19_12_MASK) >> 1;
            imm |= (inst & IMM20_20_MASK) >> 12;
            // let imm = (imm << 1) as i32;
            format!("jal x{}, {}", rd, CPU::sext(imm, 0xfff80000))
        }
        _ => panic!("CPU exception: Unrecognized opcode")
    }
}
pub fn compose_rr(opcode: u32, funct3: u32, funct7: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
    let mut inst = 0;
    inst |= opcode;
    inst |= rd << 7;
    inst |= funct3 << 12;
    inst |= rs1 << 15;
    inst |= rs2 << 20;
    inst |= funct7 << 25;
    return inst;
}
pub fn compose_imm(opcode: u32, funct3: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
    let mut inst = 0;
    inst |= opcode;
    inst |= rd << 7;
    inst |= funct3 << 12;
    inst |= rs1 << 15;
    inst |= (imm << 20) as u32;
    return inst;
}
pub fn compose_imm_f7(opcode: u32, funct3: u32, funct7: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
    let mut inst = compose_imm(opcode, funct3, rd, rs1, imm);
    inst |= funct7 << 25;
    return inst;
}
pub fn compose_s(opcode: u32, funct3: u32, rs2: u32, imm: u32, rs1: u32) -> u32 {
    let mut inst = 0;
    inst |= opcode;
    let imm_4_0 = (imm & 0x0000001f) << 7;
    inst |= imm_4_0;
    inst |= funct3 << 12;
    inst |= rs1 << 15;
    inst |= rs2 << 20;
    let imm_11_5 = (imm & 0x0000001f) << 25;
    inst |= imm_11_5;
    return inst;
}

pub fn compose_u(opcode: u32, rd: u32, imm: i32) -> u32 {
    let mut inst = 0;
    inst |= opcode;
    inst |= rd << 7;
    inst |= (imm << 12) as u32;
    return inst;
}

pub fn compose_j(opcode: u32, rd: u32, imm: u32) -> u32 {
    let mut inst = 0;
    inst |= opcode;
    inst |= rd << 7;
    let imm20 = (imm & 0x00080000) << 12;
    inst |= imm20;
    let imm10_1 = (imm & 0x000003ff) << 21;
    inst |= imm10_1;
    let imm11 = (imm & 0x00000400) << 10;
    inst |= imm11;
    let imm19_12 = (imm & 0x0007f800) << 1;
    inst |= imm19_12;
    return inst;
}
pub fn compose_b(opcode: u32, funct3: u32, rs1: u32, rs2: u32, imm: u32) -> u32 {
    let mut inst = 0;
    inst |= opcode;
    let imm11 = (imm & 0x00000400) >> 3;
    inst |= imm11;
    let imm4_1 = (imm & 0x0000000f) << 8;
    inst |= imm4_1;
    inst |= funct3 << 12;
    inst |= rs1 << 15;
    inst |= rs2 << 20;
    let imm10_5 = (imm & 0x000003f0) << 21;
    inst |= imm10_5;
    let imm12 = (imm & 0x00000800) << 20;
    inst |= imm12;
    return inst;
}
