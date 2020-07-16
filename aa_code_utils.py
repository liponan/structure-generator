def aaa2id(aa):
    table = dict(ALA=0, ARG=1, ASN=2, ASP=3, CYS=4, GLN=5, GLU=6, GLY=7, HIS=8,
                 ILE=9, LEU=10, LYS=11, MET=12, PHE=13, PRO=14, SER=15, THR=16,
                 TRP=17, TYR=18, VAL=19)
    return table[aa.upper()]


def a2id(aa):
    table = dict(A=0, R=1, N=2, D=3, C=4, E=5, Q=6, G=7, H=8,
                 I=9, L=10, K=11, M=12, F=13, P=14, S=15, T=16,
                 W=17, Y=18, V=19, X=19)
    return table[aa.upper()]


def id2a(aid):
    table = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
             "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
             "X"]
    return table[aid]


def aa3toaa1(x):
    table = dict(ala="A", arg="R", asn="N", asp="D", asx="B", cys="C", glu="E", gln="Q", glx="Z",
                 gly="G", his="H", ile="I", leu="L", lys="K", met="M", phe="F", pro="P", ser="S",
                 thr="T", trp="W", tyr="Y", val="V")
    if isinstance(x, str):
        return table[x.lower()]
    if isinstance(x, list):
        return [table[i.lower()] for i in x]


def aa1toaa3(x):
    table = dict(a="ALA", r="ARG", n="ASN", d="asp", b="asx", c="cys", e="glu", q="gln", z="glx",
                 g="gly", h="his", i="ile", l="leu", k="lys", m="met", f="phe", p="pro", s="ser",
                 t="thr", w="trp", y="tyr", v="val")
    if isinstance(x, str):
        return table[x.lower()].upper()
    if isinstance(x, list):
        return [table[i.lower()].upper() for i in x]


def aa1toidx(x):
    table = dict(A=0, R=1, N=2, D=3, C=4, E=5, Q=6, G=7, H=8, I=9,
                 L=10, K=11, M=12, F=13, P=14, S=15, T=16, W=17, Y=18, V=19)
    if isinstance(x, str):
        return table[x.upper()]
    if isinstance(x, list):
        return [table[i.upper()] for i in x]
    

def aa3toidx(x):
    return aa1toidx(aa3toaa1(x))


def idxtoaa1(idx):
    table = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
             "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
             "X"]
    if isinstance(idx, int):
        return table[idx]
    if isinstance(idx, list):
        return [table[i] for i in idx]


def idxtoaa3(idx):
    return aa1toaa3(idxtoaa1(idx))
