pub const OPCODE_MASK: u32 = 0x0000007f;
pub const RD_MASK: u32 = 0x00000f80;
pub const FUNCT3_MASK: u32 = 0x00007000;
pub const RS1_MASK: u32 = 0x000f8000;
pub const IMM12_MASK: u32 = 0xfff00000;
pub const IMM5_MASK: u32 = 0x01f00000;
pub const FUNCT7_MASK: u32 = 0xfe000000;
pub const IMM12_11_5_MASK: u32 = 0xfe000000;
pub const IMM12_4_0_MASK: u32 = 0x00000f80;
pub const RS2_MASK: u32 = 0x01f00000;
pub const IMM20_MASK: u32 = 0xfffff000;
pub const IMM20_19_12_MASK: u32 = 0x000ff000;
pub const IMM20_11_MASK: u32 = 0x00100000;
pub const IMM20_10_1_MASK: u32 = 0x7fe00000;
pub const IMM20_20_MASK: u32 = 0x80000000;
pub const SB_IMM20_11_MASK: u32 = 0x00000080;
pub const SB_IMM20_4_1_MASK: u32 = 0x00000f00;
pub const SB_IMM20_10_5_MASK: u32 = 0x7e000000;
pub const SB_IMM20_12_MASK: u32 = 0x80000000;

// CPU: includes the registers, the pc (program counter), decoupled memory source
pub struct CPU {
    pub registers: Vec<i32>,
    pub pc: u32,
    pub memory: Box<dyn Memory>
}

impl CPU {
    // Initialize stack pointer, global pointer (points to static section), and PC
    pub fn initialize(&mut self, sp: i32, gp: i32, pc: u32) {
        self.registers[2] = sp;
        self.registers[3] = gp;
        self.pc = pc;
    }
    pub fn sext(value: u32, sign_mask: u32) -> i32 {
        let sign = value & sign_mask;
        return if sign == 0 { value as i32 } else { (value | sign_mask) as i32 }
    }
    // In RISC-V, jumps/branches must have a computed address that is 4-byte aligned
    pub fn check_alignment(value: i32) -> i32 {
        if value & 0x00000003 != 0 { panic!("CPU exception: instruction misaligned") } else { value as i32 }
    }
    // Here is where the CPU starts executing..
    pub fn run(&mut self) {
        // First fetch instruction from initial PC
        let mut current_inst = self.fetch();
        while current_inst != 0 {
            // Execute current instruction, then increment PC
            self.decode_execute(current_inst);
            self.pc += 4;
            current_inst = self.fetch();
        }
    }
    // Loads instruction (word) from PC
    pub fn fetch(&mut self) -> u32 {
        self.memory.load(self.pc, 4)
    }
    pub fn decode_execute(&mut self, inst: u32) {
        // First we must match the opcode to the type of operation
        let opcode = inst & OPCODE_MASK;
        match opcode {
            // Load instructions: lb, lh, lw, lbu, lhu
            3 => {
                let rd = (inst & RD_MASK) >> 7;
                let funct3 = (inst & FUNCT3_MASK) >> 12;
                let rs1 = (inst & RS1_MASK) >> 15;
                let imm = ((inst & IMM12_MASK) as i32) >> 20;
                let address = self.registers[rs1 as usize] + imm;
                match funct3 {
                    0 => self.registers[rd as usize] = CPU::sext(self.memory.load(address as u32, 1), 0xffffff80),
                    1 => self.registers[rd as usize] = CPU::sext(self.memory.load(address as u32, 2), 0xffff8000),
                    2 => self.registers[rd as usize] = CPU::sext(self.memory.load(address as u32, 4), 0x80000000),
                    4 => self.registers[rd as usize] = self.memory.load(address as u32, 1) as i32,
                    5 => self.registers[rd as usize] = self.memory.load(address as u32, 2) as i32,
                    _ => panic!("Unrecognized funct3")
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
                let address = self.registers[rs1 as usize] + CPU::sext(imm, 0xfffff800);
                let funct3 = (inst & FUNCT3_MASK) >> 12;
                match funct3 {
                    0 => self.memory.store(address as u32, self.registers[rs2 as usize] as u32, 1),
                    1 => self.memory.store(address as u32, self.registers[rs2 as usize] as u32, 2),
                    2 => self.memory.store(address as u32, self.registers[rs2 as usize] as u32, 4),
                    _ => panic!("Unrecognized funct3")
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
                    0 => self.imm_op(rd, rs1, CPU::sext(imm12 as u32, 0xfffff800), |rs1: i32, imm: i32| { ((rs1 as i64) + (imm as i64)) as i32 }),
                    1 => self.imm_op(rd, rs1, imm5 as i32, |rs1: i32, imm: i32| { rs1 << imm }),
                    2 => self.imm_op(rd, rs1, CPU::sext(imm12 as u32, 0xfffff800), |rs1: i32, imm: i32| { if rs1 < imm { 1 } else { 0 }}),
                    3 => self.imm_op(rd, rs1, imm12 as i32, |rs1: i32, imm: i32| { if (rs1 as u32) < (imm as u32) { 1 } else { 0 }}),
                    4 => self.imm_op(rd, rs1, CPU::sext(imm12 as u32, 0xfffff800), |rs1: i32, imm: i32| { rs1 ^ imm }),
                    5 => {
                        let funct7 = (inst & FUNCT7_MASK) >> 25;
                        match funct7 {
                            0 => self.imm_op(rd, rs1, imm5 as i32, |rs1: i32, imm: i32| { ((rs1 as u32) >> imm) as i32 }),  
                            32 => self.imm_op(rd, rs1, imm5 as i32, |rs1: i32, imm: i32| { rs1 >> imm }),
                            _ => panic!("CPU exception: unrecognized funct7")
                        }
                    }
                    6 => self.imm_op(rd, rs1, CPU::sext(imm12 as u32, 0xfffff800), |rs1: i32, imm: i32| { rs1 | imm }),
                    7 => self.imm_op(rd, rs1, CPU::sext(imm12 as u32, 0xfffff800), |rs1: i32, imm: i32| { rs1 & imm }),
                    _ => panic!("CPU exception: unrecognized funct3")
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
                            0 => self.rr_op(rd, rs1, rs2, |rs1: i32, rs2: i32| { ((rs1 as i64) + (rs2 as i64)) as i32 }),
                            32 => self.rr_op(rd, rs1, rs2, |rs1: i32, rs2: i32| { ((rs1 as i64) - (rs2 as i64)) as i32 }),
                            _ => panic!("CPU exception: unrecognized funct7")
                        }
                    }
                    1 => self.rr_op(rd, rs1, rs2, |rs1: i32, rs2: i32| { rs1 << (rs2 & 0x0000001f) }),
                    2 => self.rr_op(rd, rs1, rs2, |rs1: i32, rs2: i32| { if rs1 < rs2 { 1 } else { 0 } }),
                    3 => self.rr_op(rd, rs1, rs2, |rs1: i32, rs2: i32| { if (rs1 as u32) < (rs2 as u32) { 1 } else { 0 } }),
                    4 => self.rr_op(rd, rs1, rs2, |rs1: i32, rs2: i32| { rs1 ^ rs2 }),
                    5 => {
                        match funct7 {
                            0 => self.rr_op(rd, rs1, rs2, |rs1: i32, rs2: i32| { ((rs1 as u32) >> (rs2 & 0x0000001f)) as i32 }),
                            32 => self.rr_op(rd, rs1, rs2, |rs1: i32, rs2: i32| { rs1 >> (rs2 & 0x0000001f) }),
                            _ => panic!("Unrecognized funct7")
                        }
                    }
                    6 => self.rr_op(rd, rs1, rs2, |rs1: i32, rs2: i32| { rs1 | rs2 }),
                    7 => self.rr_op(rd, rs1, rs2, |rs1: i32, rs2: i32| { rs1 & rs2 }),
                    _ => panic!("CPU exception: unrecognized funct3")
                }
            }
            // auipc 
            23 => {
                let rd = (inst & RD_MASK) >> 7;
                let imm = inst & IMM20_MASK;
                self.registers[rd as usize] = (self.pc + imm) as i32;
            }
            // lui
            55 => {
                let rd = (inst & RD_MASK) >> 7;
                let imm = inst & IMM20_MASK;
                self.registers[rd as usize] = imm as i32;
            }
            // branches: beq, bne, blt, bge, bltu, bgeu
            99 => {
                let mut imm = 0;
                imm |= (inst & SB_IMM20_4_1_MASK) >> 8;
                imm |= (inst & SB_IMM20_10_5_MASK) >> 21;
                imm |= (inst & SB_IMM20_11_MASK) << 3;
                imm |= (inst & SB_IMM20_12_MASK) >> 20;
                let imm = CPU::check_alignment(CPU::sext(imm << 1, 0xfffff000));
                let funct3 = (inst & FUNCT3_MASK) >> 12;
                let rs1 = (inst & RS1_MASK) >> 15;
                let rs2 = (inst & RS2_MASK) >> 20;
                match funct3 {
                    // We subtract immediate by 4 to simplify updating the PC
                    0 => self.pc = ((self.pc as i32) + if self.registers[rs1 as usize] == self.registers[rs2 as usize] { imm - 4 } else { 0 }) as u32,
                    1 => self.pc = ((self.pc as i32) + if self.registers[rs1 as usize] != self.registers[rs2 as usize] { imm - 4 } else { 0 }) as u32,
                    4 => self.pc = ((self.pc as i32) + if self.registers[rs1 as usize] < self.registers[rs2 as usize] { imm - 4 } else { 0 }) as u32,
                    5 => self.pc = ((self.pc as i32) + if self.registers[rs1 as usize] >= self.registers[rs2 as usize] { imm - 4 } else { 0 }) as u32,
                    6 => self.pc = ((self.pc as i32) + if (self.registers[rs1 as usize] as u32) < (self.registers[rs2 as usize] as u32) { imm - 4 } else { 0 }) as u32,
                    7 => self.pc = ((self.pc as i32) + if (self.registers[rs1 as usize] as u32) >= (self.registers[rs2 as usize] as u32) { imm - 4 } else { 0 }) as u32,
                    _ => panic!("CPU exception: unrecognized funct3")
                }
            }
            // jalr
            103 => {
                let rd = (inst & RD_MASK) >> 7;
                let rs1 = (inst & RS1_MASK) >> 15;
                let mut imm = ((inst & IMM12_MASK) as i32) >> 20;
                imm = CPU::check_alignment(imm);
                self.registers[rd as usize] = (self.pc + 4) as i32;
                self.pc = (self.registers[rs1 as usize] + imm - 4) as u32;
            }
            // jal
            111 => {
                let rd = (inst & RD_MASK) >> 7;
                let mut imm = 0;
                imm |= (inst & IMM20_10_1_MASK) >> 21;
                imm |= (inst & IMM20_11_MASK) >> 10;
                imm |= (inst & IMM20_19_12_MASK) >> 1;
                imm |= (inst & IMM20_20_MASK) >> 12;
                let mut imm = CPU::check_alignment((imm << 1) as i32);
                imm = CPU::sext(imm as u32, 0xfff00000) as i32;
                self.registers[rd as usize] = (self.pc + 4) as i32;
                self.pc = ((self.pc as i32) + imm - 4) as u32;
            }
            _ => panic!("CPU exception: Unrecognized opcode")
        }
        // Ensure x0 is always 0
        self.registers[0] = 0;
    }
    pub fn rr_op<F>(&mut self, rd: u32, rs1: u32, rs2: u32, op: F)
    where F: Fn(i32, i32) -> i32 {
        self.registers[rd as usize] = op(self.registers[rs1 as usize], self.registers[rs2 as usize]);
    }
    pub fn imm_op<F>(&mut self, rd: u32, rs1: u32, imm: i32, op: F)
    where F: Fn(i32, i32) -> i32 {
        self.registers[rd as usize] = op(self.registers[rs1 as usize], imm);
    }
}

pub trait Memory {
    fn load(&self, address: u32, size: u32) -> u32;
    fn store(&mut self, address: u32, value: u32, size: u32);
}
pub struct RAM {
    pub memory: Vec<u8>
}
impl Memory for RAM {
    fn load(&self, address: u32, size: u32) -> u32 {
        let mut value: u32 = 0;
        let mut shift = 0;
        for i in 0..size {
            value |= (self.memory[(address + i) as usize] as u32) << shift;
            shift += 8;
        }
        return value;
    }
    fn store(&mut self, address: u32, value: u32, size: u32) {
        let mut mask = 0x000000ff;
        let mut shift = 0;
        for i in 0..size {
            self.memory[(address + i) as usize] = ((value & mask) >> shift) as u8; 
            mask = mask << 8;
            shift += 8;
        }
    }
}

use std::{thread, time};
impl CPU {
    pub fn run_debug(&mut self, clock_rate: u64) {
        let mut current_inst = self.fetch();
        let mut instruction_count = 0;
        let sleep_ms = time::Duration::from_millis(1000/clock_rate);
        println!("CPU BEGIN:");
        while current_inst != 0 {
            println!("0x{:08x}: {}\n", self.pc, disassemble(current_inst));
            println!("{}\n", self.dump_registers());
            self.decode_execute(current_inst);
            self.pc += 4;
            current_inst = self.fetch();
            instruction_count += 1;
            println!("Instruction count: {}\n", instruction_count);
            println!("----------------------------------------------------------------------\n");
            thread::sleep(sleep_ms);
        }
    }
    fn dump_registers(&self) -> String {
        let mut message = String::new();
        for (index, register) in self.registers.iter().enumerate() {
            let reg_str = &format!("x{}", index);
            message.push_str(&format!("{reg_str:3}: {:10} | ", register));
            if (index+1) % 4 == 0 {
                message.push('\n');
            }
        }
        return message;
    }
}

fn disassemble(inst: u32) -> String {
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
                        _ => format!("CPU exception: unrecognized funct7")
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
                        _ => format!("CPU exception: unrecognized funct7")
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
        _ => format!("CPU exception: Unrecognized opcode")
    }
}
