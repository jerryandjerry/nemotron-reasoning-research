"""Test many hypotheses for puzzle 7192535b."""
pairs = [
    (0b01001000, 0b00000000),
    (0b00101101, 0b00000010),
    (0b10100100, 0b00001000),
    (0b00011100, 0b00000001),
    (0b00111011, 0b01100111),
    (0b11010000, 0b00001000),
    (0b10011000, 0b00010001),
    (0b01001010, 0b00000100),
    (0b11000110, 0b10001100),
]
query = 0b00100101
gold = 0b00000010

def b(x): return format(x & 0xFF, '08b')

def rotl(x,k): k%=8; return ((x<<k)|(x>>(8-k)))&0xFF if k else x&0xFF
def rotr(x,k): k%=8; return ((x>>k)|(x<<(8-k)))&0xFF if k else x&0xFF

# Try x * x mod 256, x*c for c, x+x^2 etc.
def test(name, fn):
    ok = all(fn(x) == y for x, y in pairs)
    pred = fn(query)
    if ok:
        print(f"MATCH  {name}: pred={b(pred)} gold={b(gold)} {'YES' if pred==gold else 'NO'}")

# Multiplication
for c in range(256):
    test(f"x*{c}&FF", lambda x, c=c: (x*c)&0xFF)
# Multiplication with self xor
test("x*x>>1", lambda x: (x*x)>>1 & 0xFF)
test("x*x>>2", lambda x: (x*x)>>2 & 0xFF)
test("x*x>>3", lambda x: (x*x)>>3 & 0xFF)
test("x*x>>4", lambda x: (x*x)>>4 & 0xFF)
test("x*x>>5", lambda x: (x*x)>>5 & 0xFF)
test("x*x>>6", lambda x: (x*x)>>6 & 0xFF)
test("x*x>>7", lambda x: (x*x)>>7 & 0xFF)
test("x*x>>8", lambda x: (x*x)>>8 & 0xFF)
test("x*x>>9", lambda x: (x*x)>>9 & 0xFF)
test("x*x>>10", lambda x: (x*x)>>10 & 0xFF)

# Triple-AND windows
for a,b_,c in [(0,1,2),(0,1,3),(0,2,3),(0,1,4),(1,2,3),(0,2,4),(1,2,4),(0,3,6)]:
    fn = lambda x,a=a,b_=b_,c=c: (((x<<a)&0xFF) & ((x<<b_)&0xFF) & ((x<<c)&0xFF)) & 0xFF
    test(f"x<<{a} & x<<{b_} & x<<{c}", fn)
    fn = lambda x,a=a,b_=b_,c=c: ((x>>a) & (x>>b_) & (x>>c)) & 0xFF
    test(f"x>>{a} & x>>{b_} & x>>{c}", fn)

# Multiplication & shifting
for shift in range(0, 16):
    for c in range(1, 256):
        fn = lambda x, c=c, s=shift: ((x*c) >> s) & 0xFF
        test(f"(x*{c})>>{shift}", fn)
