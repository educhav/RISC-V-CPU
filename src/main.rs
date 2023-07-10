// 0x3fbf, 0x83fd
// 1. Decode them 
// 2. Execute them
// 3. Store any result
// Registers, Memory, Immediate
// 


struct CPU {
    registers: Vec<i32>,
    memory: Vec<u8>
}

impl CPU {
    fn initialize(&mut self, sp: i32, gp: i32) {
        self.registers[2] = sp;
        self.registers[3] = gp;
    }
    fn add(&mut self, rd: usize, rs1: usize, rs2: usize) {
        self.registers[rd] = self.registers[rs1] + self.registers[rs2];
    }
    fn addi(&mut self, rd: usize, rs1: usize, imm: i32) {
        self.registers[rd] = self.registers[rs1] + imm;
    }
}

fn decode_cycle(mut cpu: CPU, mut instructions: Vec<i32>) {
    for mut inst in instructions {
        let opcode = inst % (2 << 6);
        inst = inst >> 7;

        match opcode {
            19 => {
                let rd = inst % (2 << 4);
                inst = inst >> 5;
                let funct3 = inst % (2 << 2);
                inst = inst >> 3;
                match funct3 {
                    0 => {
                        let rs1 = inst % (2 << 4);
                        inst = inst >> 5;
                        let imm = inst;
                        cpu.addi(rd.try_into().unwrap(), rs1.try_into().unwrap(), imm);
                    }
                    // TODO: CPU exception
                    _ => ()
                }
            }
            51 => {
            }
            // TODO: CPU exception
            _ => ()
        }
        let rd = inst % (2 << 4);
        println!("opcode: {}, rd: {}", opcode, rd);
    }
}

fn main() {
    let mut cpu = CPU {
        registers: vec![0; 32],
        memory: vec![0; 2 << 31]
    };
    cpu.initialize(0x7ffffff0, 0x10000000);

    let instructions = vec![0b00000000000100000000000010010011, 0b00000000001000001000000010110011];

    decode_cycle(cpu, instructions);

    println!("{:08b}", -1);



}
