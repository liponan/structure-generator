def aaa2id(aa):
    table = dict(ALA=0, ARG=1, ASN=2, ASP=3, CYS=4, GLN=5, GLU=6, GLY=7, HIS=8,
                 ILE=9, LEU=10, LYS=11, MET=12, PHE=13, PRO=14, SER=15, THR=16,
                 TRP=17, TYR=18, VAL=19)
    return table[aa.upper()]


def a2id(aa):
    table = dict(A=0, R=1, N=2, D=3, C=4, Q=5, E=6, G=7, H=8,
                 I=9, L=10, K=11, M=12, F=13, P=14, S=15, T=16,
                 W=17, Y=18, V=19, X=19)
    return table[aa.upper()]


def id2a(aid):
    table = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
             "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
             "X"]
    return table[aid]


def aa3toaa1(x):
    table = dict(ala="A", arg="R", asn="N", asp="D", asx="B", cys="C", gln="Q", glu="E", glx="Z",
                 gly="G", his="H", ile="I", leu="L", lys="K", met="M", phe="F", pro="P", ser="S",
                 thr="T", trp="W", tyr="Y", val="V", unk="X", ukn="X", sec="U", pyl="O")
    if isinstance(x, str):
        return table[x.lower()]
    if isinstance(x, list):
        return [table[i.lower()] for i in x]


def aa1toaa3(x):
    table = dict(a="ALA", r="ARG", n="ASN", d="asp", b="asx", c="cys", q="gln", e="glu", z="glx",
                 g="gly", h="his", i="ile", l="leu", k="lys", m="met", f="phe", p="pro", s="ser",
                 t="thr", w="trp", y="tyr", v="val", u="sec", x="unk", o="pyl")
    if isinstance(x, str):
        return table[x.lower()].upper()
    if isinstance(x, list):
        return [table[i.lower()].upper() for i in x]


def aa1toidx(x):
    table = dict(A=0, R=1, N=2, D=3, C=4, Q=5, E=6, G=7, H=8, I=9,
                 L=10, K=11, M=12, F=13, P=14, S=15, T=16, W=17, Y=18, V=19)
    if isinstance(x, str):
        return table[x.upper()]
    if isinstance(x, list):
        return [table[i.upper()] for i in x]
    

def aa3toidx(x):
    return aa1toidx(aa3toaa1(x))


def idxtoaa1(idx):
    table = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
             "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
             "X"]
    if isinstance(idx, int):
        return table[idx]
    if isinstance(idx, list):
        return [table[i] for i in idx]


def idxtoaa3(idx):
    return aa1toaa3(idxtoaa1(idx))


def rot_to_aa_idx(rot_label):
    table = dict(ala_1="ala", arg_1="arg", arg_10="arg", arg_11="arg", arg_12="arg", arg_13="arg", arg_14="arg",
                 arg_15="arg", arg_16="arg", arg_17="arg", arg_18="arg", arg_19="arg", arg_2="arg", arg_20="arg",
                 arg_21="arg", arg_22="arg", arg_23="arg", arg_24="arg", arg_25="arg", arg_26="arg", arg_27="arg",
                 arg_28="arg", arg_29="arg", arg_3="arg", arg_30="arg", arg_31="arg", arg_32="arg", arg_33="arg",
                 arg_34="arg", arg_4="arg", arg_5="arg", arg_6="arg", arg_7="arg", arg_8="arg", arg_9="arg",
                 asn_1="asn", asn_2="asn", asn_3="asn", asn_4="asn", asn_5="asn", asn_6="asn", asn_7="asn", asp_1="asp",
                 asp_2="asp", asp_3="asp", asp_4="asp", asp_5="asp", cys_1="cys", cys_2="cys", cys_3="cys", gln_1="gln",
                 gln_2="gln", gln_3="gln", gln_4="gln", gln_5="gln", gln_6="gln", gln_7="gln", gln_8="gln", gln_9="gln",
                 glu_1="glu", glu_2="glu", glu_3="glu", glu_4="glu", glu_5="glu", glu_6="glu", glu_7="glu", glu_8="glu",
                 gly_1="gly", his_1="his", his_2="his", his_3="his", his_4="his", his_5="his", his_6="his", his_7="his",
                 his_8="his", ile_1="ile", ile_2="ile", ile_3="ile", ile_4="ile", ile_5="ile", ile_6="ile", ile_7="ile",
                 leu_1="leu", leu_2="leu", leu_3="leu", leu_4="leu", leu_5="leu", lys_1="lys", lys_10="lys",
                 lys_11="lys", lys_12="lys", lys_13="lys", lys_14="lys", lys_15="lys", lys_16="lys", lys_17="lys",
                 lys_18="lys", lys_19="lys", lys_2="lys", lys_20="lys", lys_21="lys", lys_22="lys", lys_23="lys",
                 lys_24="lys", lys_25="lys", lys_26="lys", lys_27="lys", lys_3="lys", lys_4="lys", lys_5="lys",
                 lys_6="lys", lys_7="lys", lys_8="lys", lys_9="lys", met_1="met", met_10="met", met_11="met",
                 met_12="met", met_13="met", met_2="met", met_3="met", met_4="met", met_5="met", met_6="met",
                 met_7="met", met_8="met", met_9="met", phe_1="phe", phe_2="phe", phe_3="phe", phe_4="phe", phe_5="phe",
                 phe_6="phe", phe_7="phe", phe_8="phe", pro_1="pro", pro_2="pro", ser_1="ser", ser_2="ser", ser_3="ser",
                 thr_1="thr", thr_2="thr", thr_3="thr", trp_1="trp", trp_2="trp", trp_3="trp", trp_4="trp", trp_5="trp",
                 trp_6="trp", trp_7="trp", tyr_1="tyr", tyr_2="tyr", tyr_3="tyr", tyr_4="tyr", tyr_5="tyr", tyr_6="tyr",
                 tyr_7="tyr", tyr_8="tyr", val_1="val", val_2="val", val_3="val", ukn_1="unk", unk_1="unk")
    return aa3toidx(table[rot_label.lower()])


def rot_to_idx(rot_label):
    table = dict(ala_1=0, arg_1=1, arg_10=2, arg_11=3, arg_12=4, arg_13=5, arg_14=6, arg_15=7, arg_16=8, arg_17=9,
                 arg_18=10, arg_19=11, arg_2=12, arg_20=13, arg_21=14, arg_22=15, arg_23=16, arg_24=17, arg_25=18,
                 arg_26=19, arg_27=20, arg_28=21, arg_29=22, arg_3=23, arg_30=24, arg_31=25, arg_32=26, arg_33=27,
                 arg_34=28, arg_4=29, arg_5=30, arg_6=31, arg_7=32, arg_8=33, arg_9=34, asn_1=35, asn_2=36, asn_3=37,
                 asn_4=38, asn_5=39, asn_6=40, asn_7=41, asp_1=42, asp_2=43, asp_3=44, asp_4=45, asp_5=46, cys_1=47,
                 cys_2=48, cys_3=49, gln_1=50, gln_2=51, gln_3=52, gln_4=53, gln_5=54, gln_6=55, gln_7=56, gln_8=57,
                 gln_9=58, glu_1=59, glu_2=60, glu_3=61, glu_4=62, glu_5=63, glu_6=64, glu_7=65, glu_8=66, gly_1=67,
                 his_1=68, his_2=69, his_3=70, his_4=71, his_5=72, his_6=73, his_7=74, his_8=75, ile_1=76, ile_2=77,
                 ile_3=78, ile_4=79, ile_5=80, ile_6=81, ile_7=82, leu_1=83, leu_2=84, leu_3=85, leu_4=86, leu_5=87,
                 lys_1=88, lys_10=89, lys_11=90, lys_12=91, lys_13=92, lys_14=93, lys_15=94, lys_16=95, lys_17=96,
                 lys_18=97, lys_19=98, lys_2=99, lys_20=100, lys_21=101, lys_22=102, lys_23=103, lys_24=104, lys_25=105,
                 lys_26=106, lys_27=107, lys_3=108, lys_4=109, lys_5=110, lys_6=111, lys_7=112, lys_8=113, lys_9=114,
                 met_1=115, met_10=116, met_11=117, met_12=118, met_13=119, met_2=120, met_3=121, met_4=122, met_5=123,
                 met_6=124, met_7=125, met_8=126, met_9=127, phe_1=128, phe_2=129, phe_3=130, phe_4=131, phe_5=132,
                 phe_6=133, phe_7=134, phe_8=135, pro_1=136, pro_2=137, ser_1=138, ser_2=139, ser_3=140, thr_1=141,
                 thr_2=142, thr_3=143, trp_1=144, trp_2=145, trp_3=146, trp_4=147, trp_5=148, trp_6=149, trp_7=150,
                 tyr_1=151, tyr_2=152, tyr_3=153, tyr_4=154, tyr_5=155, tyr_6=156, tyr_7=157, tyr_8=158, val_1=159,
                 val_2=160, val_3=161, ukn_1=162, unk_1=162)
    if isinstance(rot_label, str):
        return table[rot_label.lower()]
    if isinstance(rot_label, list):
        return [table[i.lower()] for i in rot_label]
