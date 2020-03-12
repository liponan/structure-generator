def aa3toaa1(x):
    table = dict(ala="A", arg="R", asn="N", asp="D", asx="B", cys="C", glu="E", gln="Q", glx="Z",
                gly="G", his="H", ile="I", leu="L", lys="K", met="M", phe="F", pro="P", ser="S",
                thr="T", trp="W", tyr="Y", val="V")
    if isinstance(x, str):
        return table[x.lower()]
    if isinstance(x, list):
        return [table[i.lower()] for i in x]
    
    
def aa1toidx(x):
    table = dict(A=0, R=1, N=2, D=3, C=4, E=5, Q=6, G=7, H=8, I=9,
           L=10, K=11, M=12, F=13, P=14, S=15, T=16, W=17, Y=18, V=19)
    if isinstance(x, str):
        return table[x.upper()]
    if isinstance(x, list):
        return [table[i.upper()] for i in x]
    

def aa3toidx(x):
    return aa1toidx(aa3toaa1(x))