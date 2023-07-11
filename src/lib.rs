// Masks we need for extracting bits from parts of the instructions
const OPCODE_MASK: u32 = 0x0000007f;
const RD_MASK: u32 = 0x00000f80;
const FUNCT3_MASK: u32 = 0x00007000;
const RS1_MASK: u32 = 0x000f8000;
const IMM12_MASK: u32 = 0xfff00000;
const IMM5_MASK: u32 = 0x01f00000;
const FUNCT7_MASK: u32 = 0xf7000000;

pub struct CPU {
    pub registers: Vec<i32>,
    pub memory: Box<dyn Memory>
}

pub trait Memory {
    fn load(&self, address: u32, size: u32) -> i32;
    fn store(&self, address: u32, value: i32, size: u32);
}

pub struct RAM {
    pub memory: Vec<u8>
}

impl Memory for RAM {
    fn load(&self, address: u32, size: u32) -> i32 {
        let mut value: i32 = 0;
        let mut shift = 0;
        for i in 0..size {
            value |= (self.memory[(address + i) as usize] as u32) << shift;
            shift += 8;
        }
        return value;
    }
    fn store(&self, address: u32, value: i32, size: u32) {
        let shift = 0;
        for i in 0..size {
            self.memory[(address + i) as usize] << shift;
        }
    }
}


impl CPU {
    // Initialize stack pointer, global pointer, and PC
    pub fn initialize(&mut self, sp: i32, gp: i32) {
        self.registers[2] = sp;
        self.registers[3] = gp;
    }
    pub fn decode_execute(&mut self, inst: u32) {
        // OPCODE is going to be last 7 bits.. 
        let opcode = inst & OPCODE_MASK;
        match opcode {
            // Load instructions: lb, lh, lw, lbu, lhu
            3 => {
                let rd = (inst & RD_MASK) >> 7;
                let funct3 = (inst & FUNCT3_MASK) >> 12;
                match funct3 {
                    0 => {
                        let rs1 = (inst & RS1_MASK) >> 15;
                        let imm = ((inst & IMM12_MASK) as i32) >> 20;
                        let address = self.registers[rs1 as usize] + imm;
                        let loaded_value = self.memory.load(address as u32, 1);
                        self.registers[rd as usize] = loaded_value;
                    }
                    _ => panic!("Unrecognized funct3")
                }

            }
            
            // Immediate ops: addi, slli, slti, sltiu, xori, srli, srai, ori, andi
            19 => {
                let rd = (inst & RD_MASK) >> 7;
                let funct3 = (inst & FUNCT3_MASK) >> 12;
                match funct3 {
                    0 => {
                        let rs1 = (inst & RS1_MASK) >> 15;
                        let imm = ((inst & IMM12_MASK) as i32) >> 20;
                        self.addi(rd.try_into().unwrap(), rs1.try_into().unwrap(), imm.try_into().unwrap());
                    }
                    1 => {
                        let rs1 = (inst & RS1_MASK) >> 15;
                        let imm = (inst & IMM5_MASK) >> 20;
                        self.slli(rd.try_into().unwrap(), rs1.try_into().unwrap(), imm.try_into().unwrap());
                    }
                    2 => {
                        let rs1 = (inst & RS1_MASK) >> 15;
                        let imm = ((inst & IMM12_MASK) as i32) >> 20;
                        self.slti(rd.try_into().unwrap(), rs1.try_into().unwrap(), imm.try_into().unwrap());
                    }
                    3 => {
                        let rs1 = (inst & RS1_MASK) >> 15;
                        let imm = (inst & IMM12_MASK) >> 20;
                        self.slti(rd.try_into().unwrap(), rs1.try_into().unwrap(), imm.try_into().unwrap());
                    }
                    4 => {
                        let rs1 = (inst & RS1_MASK) >> 15;
                        let imm = ((inst & IMM12_MASK) as i32) >> 20;
                        self.xori(rd.try_into().unwrap(), rs1.try_into().unwrap(), imm.try_into().unwrap());
                    }
                    5 => {
                        let rs1 = (inst & RS1_MASK) >> 15;
                        let imm = (inst & IMM5_MASK) >> 20;
                        let funct7 = (inst & FUNCT7_MASK) >> 25;
                        let arithmetic: bool;
                        match funct7 {
                            0 => {
                                arithmetic = false;
                            }
                            32 => {
                                arithmetic = true;
                            }
                            _ => panic!("CPU exception: Unrecognized funct7")
                        }

                        self.srli(rd.try_into().unwrap(), rs1.try_into().unwrap(), imm.try_into().unwrap(), arithmetic);
                    }
                    6 => {
                        let rs1 = (inst & RS1_MASK) >> 15;
                        let imm = ((inst & IMM12_MASK) as i32) >> 20;
                        self.ori(rd.try_into().unwrap(), rs1.try_into().unwrap(), imm.try_into().unwrap());
                    }
                    7 => {
                        let rs1 = (inst & RS1_MASK) >> 15;
                        let imm = ((inst & IMM12_MASK) as i32) >> 20;
                        self.andi(rd.try_into().unwrap(), rs1.try_into().unwrap(), imm.try_into().unwrap());

                    }
                    _ => panic!("CPU exception: unrecognized funct3")
                }
            }
            // RR ops: add, sub, sll, slt, sltu, xor, srl, sra, or, and, lui
            51 => {
            }
            _ => panic!("CPU exception: Unrecognized opcode")
        }
    }
    pub fn add(&mut self, rd: usize, rs1: usize, rs2: usize) {
        self.registers[rd] = self.registers[rs1] + self.registers[rs2];
    }
    pub fn addi(&mut self, rd: usize, rs1: usize, imm: i32) {
        self.registers[rd] = self.registers[rs1] + imm;
    }
    pub fn slli(&mut self, rd: usize, rs1: usize, imm: i32) {
        self.registers[rd] = self.registers[rs1] << imm;
    }
    pub fn slti(&mut self, rd: usize, rs1: usize, imm: i32) {
        self.registers[rd] = if self.registers[rs1] < imm { 1 } else { 0 };
    }
    pub fn xori(&mut self, rd: usize, rs1: usize, imm: i32) {
        self.registers[rd] = self.registers[rs1] ^ imm;
    }
    pub fn srli(&mut self, rd: usize, rs1: usize, imm: i32, arithmetic: bool) {
        if arithmetic {
            self.registers[rd] = self.registers[rs1] >> imm;
            return;
        }
        self.registers[rd] = ((self.registers[rs1] as u32) >> imm) as i32;
    }
    pub fn ori(&mut self, rd: usize, rs1: usize, imm: i32) {
        self.registers[rd] = self.registers[rs1] | imm;
    }
    pub fn andi(&mut self, rd: usize, rs1: usize, imm: i32) {
        self.registers[rd] = self.registers[rs1] & imm;
    }
}


#[cfg(test)]
mod tests {
    fn setup_cpu() -> super::CPU {
        let mut cpu = super::CPU {
            registers: vec![0; 32],
            memory: Box::new(super::RAM {
                memory: vec![0; 2 << 31]
            })
        };
        cpu.initialize(0x7ffffff0, 0x10000000);
        return cpu;
    }

    // helper to compose immediate instructions
    fn compose_imm(opcode: u32, mut funct3: u32, mut rd: u32, mut rs1: u32, mut imm: i32) -> u32 {
        let mut inst = 0;
        inst |= opcode;
        rd = rd << 7;
        inst |= rd;
        funct3 = funct3 << 12;
        inst |= funct3;
        rs1 = rs1 << 15;
        inst |= rs1;
        imm = imm << 20;
        inst |= imm as u32;
        return inst;
    }
    fn compose_imm_f7(opcode: u32, funct3: u32, mut funct7: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
        let mut inst = compose_imm(opcode, funct3, rd, rs1, imm);
        funct7 = funct7 << 25;
        inst |= funct7;
        return inst;
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
        cpu.initialize(0x7ffffff0, 0x10000000);
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
        assert_eq!(2047, cpu.registers[12]);

        // sltiu x12, x12, 2096
        let inst = compose_imm(19, 3, 12, 12, 2096);
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
}
