const OPCODE_MASK: u32 = 0x0000007f;
const RD_MASK: u32 = 0x00000f80;
const FUNCT3_MASK: u32 = 0x00007000;
const RS1_MASK: u32 = 0x000f8000;
const IMM12_MASK: u32 = 0xfff00000;
const IMM5_MASK: u32 = 0x01f00000;
const FUNCT7_MASK: u32 = 0xfe000000;
const IMM12_11_5_MASK: u32 = 0xfe000000;
const IMM12_4_0_MASK: u32 = 0x00000f80;
const RS2_MASK: u32 = 0x01f00000;
const IMM20_MASK: u32 = 0xfffff000;
const IMM20_19_12_MASK: u32 = 0x000ff000;
const IMM20_11_MASK: u32 = 0x00100000;
const IMM20_10_1_MASK: u32 = 0x7fe00000;
const IMM20_20_MASK: u32 = 0x80000000;
const SB_IMM20_11_MASK: u32 = 0x00000080;
const SB_IMM20_4_1_MASK: u32 = 0x00000f00;
const SB_IMM20_10_5_MASK: u32 = 0x7e000000;
const SB_IMM20_12_MASK: u32 = 0x80000000;

pub struct CPU {
    pub registers: Vec<i32>,
    pub pc: u32,
    pub memory: Box<dyn Memory>
}

impl CPU {
    pub fn initialize(&mut self, sp: i32, gp: i32, pc: u32) {
        self.registers[2] = sp;
        self.registers[3] = gp;
        self.pc = pc;
    }
    pub fn sext(value: u32, sign_mask: u32) -> i32 {
        let sign = value & sign_mask;
        return if sign == 0 { value as i32 } else { (value | sign_mask) as i32 }
    }
    pub fn check_alignment(value: i32) -> i32 {
        if value & 0x00000003 != 0 { panic!("CPU exception: instruction misaligned") } else { value as i32 }
    }
    pub fn run(&mut self) {
        let current_inst = self.fetch();
        while current_inst != 0 {
            self.decode_execute(current_inst);
            self.pc += 4;
        }
    }
    pub fn fetch(&mut self) -> u32 {
        self.memory.load(self.pc, 4)
    }
    pub fn decode_execute(&mut self, inst: u32) {
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
                let address = self.registers[rs1 as usize] + (imm as i32);
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
                let imm = CPU::sext(imm << 1, 0xfffff000);
                let funct3 = (inst & FUNCT3_MASK) >> 12;
                let rs1 = (inst & RS1_MASK) >> 15;
                let rs2 = (inst & RS2_MASK) >> 20;
                match funct3 {
                    0 => self.pc = if self.registers[rs1 as usize] == self.registers[rs2 as usize] { ((self.pc as i32) + CPU::check_alignment(imm) - 4) as u32 } else { self.pc },
                    1 => self.pc = if self.registers[rs1 as usize] != self.registers[rs2 as usize] { ((self.pc as i32) + CPU::check_alignment(imm) - 4) as u32 } else { self.pc },
                    4 => self.pc = if self.registers[rs1 as usize] < self.registers[rs2 as usize] { ((self.pc as i32) + CPU::check_alignment(imm) - 4) as u32 } else { self.pc },
                    5 => self.pc = if self.registers[rs1 as usize] >= self.registers[rs2 as usize] { ((self.pc as i32) + CPU::check_alignment(imm) - 4) as u32 } else { self.pc },
                    6 => self.pc = if (self.registers[rs1 as usize] as u32) < (self.registers[rs2 as usize] as u32) { ((self.pc as i32) + CPU::check_alignment(imm) - 4) as u32 } else { self.pc },
                    7 => self.pc = if (self.registers[rs1 as usize] as u32) >= (self.registers[rs2 as usize] as u32) { ((self.pc as i32) + CPU::check_alignment(imm) - 4) as u32 } else { self.pc },
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

#[cfg(test)]
mod tests {
    fn setup_cpu() -> super::CPU {
        let mut cpu = super::CPU {
            registers: vec![0; 32],
            pc: 0,
            memory: Box::new(super::RAM {
                memory: vec![0; 2 << 31]
            })
        };
        cpu.initialize(0x7ffffff0, 0x10000000, 0x00400000);
        return cpu;
    }
    fn compose_rr(opcode: u32, funct3: u32, funct7: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
        let mut inst = 0;
        inst |= opcode;
        inst |= rd << 7;
        inst |= funct3 << 12;
        inst |= rs1 << 15;
        inst |= rs2 << 20;
        inst |= funct7 << 25;
        return inst;
    }
    fn compose_imm(opcode: u32, funct3: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
        let mut inst = 0;
        inst |= opcode;
        inst |= rd << 7;
        inst |= funct3 << 12;
        inst |= rs1 << 15;
        inst |= (imm << 20) as u32;
        return inst;
    }
    fn compose_imm_f7(opcode: u32, funct3: u32, funct7: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
        let mut inst = compose_imm(opcode, funct3, rd, rs1, imm);
        inst |= funct7 << 25;
        return inst;
    }
    fn compose_s(opcode: u32, funct3: u32, rs2: u32, imm: u32, rs1: u32) -> u32 {
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

    fn compose_u(opcode: u32, rd: u32, imm: i32) -> u32 {
        let mut inst = 0;
        inst |= opcode;
        inst |= rd << 7;
        inst |= (imm << 12) as u32;
        return inst;
    }

    fn compose_j(opcode: u32, rd: u32, imm: u32) -> u32 {
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
    fn compose_b(opcode: u32, funct3: u32, rs1: u32, rs2: u32, imm: u32) -> u32 {
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
