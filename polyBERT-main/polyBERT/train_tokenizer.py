import sentencepiece as spm

elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

small_elements = [i.lower() for i in elements]

# els = ['C', 'Br', 'Cl', 'N', 'O', 'S', 'P', 'F', 'I', 'b', 'c', 'n', 'o', 's', 'p', "H", "Si"]
special_tokens =[
    "<pad>",
    "<mask>",
    "[*]",
    "(", ")", "=", "@", "#",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "-", "+",
    "/", "\\",
    "%", "[", "]",
]

special_tokens += elements + small_elements

# unk, .. tokens are already defined
# The file generated_polymer_smiles.txt needs to contain the 100m PSMILES strings. One per line.
spm.SentencePieceTrainer.train(input='generated_polymer_smiles.txt',
                               model_prefix='spm',
                               vocab_size=265,
                               input_sentence_size=5_000_000,
                            #    shuffle_input_sentence=True, # data set is already shuffled
                               character_coverage=1,
                               user_defined_symbols=special_tokens,
                               )



# E.g., run with 
# poetry run python train_tokenizer.py