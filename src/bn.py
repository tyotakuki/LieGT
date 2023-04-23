import numpy as np
import sympy as sympy
import itertools
import functools
import operator
import copy
from hashlib import sha1

class RootDatBn:
    __epsilon_list = {}
    
    def __init__(self, n):
        self.rank = n
        self.roots = self.__RootsBn()
        self.simple_roots = self.__RootsBnSimple()
        self.coroots = self.__CorootsBn()
        self.simple_coroots = self.__CorootsBnSimple()
        self.cartan_matrix = self.__CartanMatrixBn()
        self.fundamental_weights = self.__FundamentalWeights()
        self.root_strings = self.__RootsBnSimpleBasis()
        self.positive_root_strings = [x for x in self.root_strings if sum(x) > 0]
        self.highest_root_string = (self.positive_root_strings)[-1]
        self.highest_root = sympy.Matrix(self.highest_root_string).T * sympy.Matrix(self.simple_roots)
        self.root_order = self.__RootsBnOrder()
        self.positive_root_order = self.__RootsBnPosOrder()

    
    def __VectorByEntry(self, lst, ln):
    #List format: [[entry1, val1],[entry2, val2]]
        res = sympy.zeros(1, ln)
        for dat in lst:
            res[dat[0]] = dat[1]
        return res
    
    def __RootsBn(self):
        n = self.rank
        res = []
        for j in range(0, n):
            res.append(self.__VectorByEntry([[j, 1]], n))
            res.append(self.__VectorByEntry([[j, -1]], n))
            for i in range(0, j):
                res.append(self.__VectorByEntry([[i, 1], [j, 1]], n))
                res.append(self.__VectorByEntry([[i, -1], [j, -1]], n))
                res.append(self.__VectorByEntry([[i, 1], [j, -1]], n))
                res.append(self.__VectorByEntry([[i, -1], [j, 1]], n))
        res.sort(key = tuple)
        return res
    
    def __RootsBnSimple(self):
        n = self.rank
        res = []
        for j in range(0, n-1):
            res.append(self.__VectorByEntry([[j, 1], [j+1, -1]], n))
        res.append(self.__VectorByEntry([[n-1, 1]], n))
        return res
    
    def __CorootsBn(self):
        return [(2*x/(x.dot(x))) for x in self.roots]
    
    def __CorootsBnSimple(self):
        return [(2*x/(x.dot(x))) for x in self.simple_roots]
    
    def __CorootsInnerProduct(self):
        return sympy.Matrix([[a.dot(b) for a in self.simple_coroots] for b in self.simple_coroots])
    
    def __CartanMatrixBn(self):
        return sympy.Matrix([[a.dot(b) for a in self.simple_roots] for b in self.simple_coroots])
    
    def __FundamentalWeights(self):
        crt = self.simple_coroots
        ipinv = sympy.Matrix(self.__CorootsInnerProduct()).inv()
        fw = ((sympy.Matrix(crt).T)*ipinv).T.tolist()
        return [sympy.Matrix([[el for el in vec]]) for vec in fw]
   
    def __RootsBnSimpleBasis(self):
        mat = (sympy.Matrix(self.roots) * sympy.Matrix(self.simple_roots).inv())
        res = [list(mat.row(k)) for k in range(0, mat.rows)]
        res.sort(key = sum)
        return res
    
    def __RootsBnOrder(self):
        allroots = self.root_strings
        allroots.sort(key=sum)
        return dict(zip(map(tuple,allroots), range(0,len(allroots))))
    
    def __RootsBnPosOrder(self):
        allroots = self.positive_root_strings
        allroots.sort(key=sum)
        return dict(zip(map(tuple,allroots), range(0,len(allroots))))
    
    def get_p(self, rtstr1, rtstr2):
        p = 0
        rtstr1 = sympy.Matrix(rtstr1)
        rtstr2 = sympy.Matrix(rtstr2)
        allroots = map(lambda x: sympy.Matrix(x), self.root_strings)
        while rtstr2 - (p + 1) * rtstr1 in allroots:
            p += 1
        return p
    
    def __GetExtraspecialBn(self, alpha, beta):
        xi = [x + y for x,y in zip(alpha,beta)]
        rts = list(self.positive_root_order.keys())
        if tuple(xi) in rts:
            z = 0
            while tuple(sympy.Matrix(xi) - sympy.Matrix(rts[z])) not in rts:
                z += 1
            return [list(rts[z]), list(sympy.Matrix(xi) - sympy.Matrix(rts[z]))]
        else:
            return None

    def __rtlensq(self, rt):
        vec = sympy.Matrix([rt]) * sympy.Matrix(self.simple_roots)
        return vec.dot(vec)
    
    def get_epsilon(self, alpha, beta):
        if tuple(alpha+beta) in self.__epsilon_list.keys():
            return self.__epsilon_list[tuple(alpha+beta)]
        else:
            xi = list(sympy.Matrix(alpha) + sympy.Matrix(beta))
            allroots = self.root_strings
            ord = self.root_order
            posrt = list(ord.keys())
            if xi in allroots:
                if sum(beta) < 0:
                    val = -self.get_epsilon(list(-sympy.Matrix(alpha)),list(-sympy.Matrix(beta)))

                else:
                    if sum(alpha) < 0:
                        if ord[tuple(-sympy.Matrix(alpha))] < ord[tuple(beta)]:
                            val = self.get_epsilon(list(sympy.Matrix(alpha) + sympy.Matrix(beta)),list(-sympy.Matrix(alpha)))

                        else:
                            val = self.get_epsilon(beta, list(- sympy.Matrix(alpha) - sympy.Matrix(beta)))

                    else:
                        if ord[tuple(alpha)] > ord[tuple(beta)]:
                            val = -self.get_epsilon(beta, alpha)

                        else:
                            exsp = self.__GetExtraspecialBn(alpha, beta)
                            if exsp == [alpha, beta]:
                                val = 1

                            else:
                                if list(sympy.Matrix(beta) - sympy.Matrix(exsp[0])) in allroots:
                                    t1 = self.__rtlensq(list(sympy.Matrix(beta) - sympy.Matrix(exsp[0])))/self.__rtlensq(beta) * (self.get_p(exsp[0], list(sympy.Matrix(beta) - sympy.Matrix(exsp[0])))+1)*self.get_epsilon(exsp[0], list(sympy.Matrix(beta) - sympy.Matrix(exsp[0])))*(self.get_p(alpha, list(sympy.Matrix(exsp[1])- sympy.Matrix(alpha)))+1)*self.get_epsilon(alpha, list(sympy.Matrix(exsp[1])- sympy.Matrix(alpha)))
                                else:
                                    t1 = 0
                                if list(sympy.Matrix(alpha) - sympy.Matrix(exsp[0])) in allroots:
                                    t2 = self.__rtlensq(list(sympy.Matrix(alpha) - sympy.Matrix(exsp[0])))/self.__rtlensq(alpha) * (self.get_p(exsp[0], list(sympy.Matrix(alpha) - sympy.Matrix(exsp[0])))+1)*self.get_epsilon(exsp[0], list(sympy.Matrix(alpha) - sympy.Matrix(exsp[0])))*(self.get_p(beta, list(sympy.Matrix(exsp[1]) - sympy.Matrix(beta)))+1)*self.get_epsilon(beta, list(sympy.Matrix(exsp[1]) - sympy.Matrix(beta)))
                                else:
                                    t2 = 0
                                if t1 - t2 >= 0:
                                    val = 1
                                else:
                                    val = -1
            else:
                val = 0
            self.__epsilon_list[tuple(alpha+beta)] = val
            return val

        
class GTPattern:
    def __init__(self, patt):
        self.pattern = patt
    
    def raise_pattern(self, k, i):
        self.pattern.reverse()
        self.pattern[k-1][i-1] += 1
        self.pattern.reverse()
    
    def raised(self, k, i):
        res = copy.deepcopy(self.pattern)
        res.reverse()
        res[k-1][i-1] += 1
        res.reverse()
        return res
        
    def lower_pattern(self, k, i):
        self.pattern.reverse()
        self.pattern[k-1][i-1] -= 1
        self.pattern.reverse()
    
    def lowered(self, k, i):
        res = copy.deepcopy(self.pattern)
        res.reverse()
        res[k-1][i-1] -= 1
        res.reverse()
        return res
    
    
    def get_lambda(self, k, i):
        res = copy.deepcopy(self.pattern)
        res.reverse()
        return res[k-1][i-1]


    def get_l(self, k, i):
        res = copy.deepcopy(self.pattern)
        res.reverse()
        if k%2 == 0:
            return res[k-1][i-1] + sympy.Rational(k,2) - i + 1
        else:
            return res[k-1][i-1] + sympy.Rational(k+1,2) - i
    
        
        
class RepBn:
    def __init__(self, hwt, denom):
        self.highest_weight = list(sympy.Matrix(hwt)/denom)
        self.root_data = RootDatBn(len(hwt))
        self.Gelfand_Tsetlin_patterns = self.__GenerateAllPatterns(self.highest_weight)
        self.patterns = [GTPattern(x) for x in self.Gelfand_Tsetlin_patterns]
        self.pattern_order = dict(zip([self.__hsh(patt.pattern) for patt in self.patterns], range(0,len(self.patterns))))
        self.root_vectors = {}
        
    def __range_halfint(self, *args):
        assert args[0].denominator == args[1].denominator, "should both be integer or half integer"
        if args[0].denominator == 1:
            return list(range(*args))
        else:
            if len(args) == 3:
                return list(sympy.Matrix(range(*list(sympy.Matrix(args) * 2)))/2)
            elif len(args) == 2:
                return list(sympy.Matrix(range(*(list(sympy.Matrix(args) * 2) + [2])))/2)
    
    def __FirstRow(self, rw):
        res = list(-sympy.Matrix(rw))
        res.reverse()
        return res
    
    def __NextRow(self, rw):
        newrw = copy.deepcopy(rw)
        newrw[0] = -abs(newrw[0])
        lst = [list(self.__range_halfint(newrw[r+1], newrw[r]+1)) for r in range(0, len(newrw)-1)]
        return [list(x) for x in itertools.product(*lst)]
    
    def __NextAuxRow(self, rw):
        if rw[0].denominator == 1:
            rw = [0]+rw
        elif rw[0].denominator == 2:
            rw = [sympy.Rational(1,2)] + rw
        lst = [list(self.__range_halfint(rw[r+1], rw[r]+1)) for r in range(0, len(rw)-1)]
        prep = [list(x) for x in itertools.product(*lst)]
        res0 = copy.deepcopy(prep)
        for newrow in prep:
            if newrow[0] != 0:
                new1 = copy.deepcopy(newrow)
                new1[0] = - new1[0]
                res0 += [new1]
        res = []
        [res.append(x) for x in res0 if x not in res]
        return res
    
    def __NextCallback(self, patt, fun):
        lstrow = patt[-1]
        nxt = fun(lstrow)
        return [patt+[x] for x in nxt]
    
    def __NextPattern(self, patt):
        nxtaux = self.__NextCallback(patt, self.__NextAuxRow)
        nxt = []
        for ptt in nxtaux:
            nxt = nxt + self.__NextCallback(ptt, self.__NextRow)
        return nxt
    
    def __GenerateAllPatterns(self, wt):
        this = self.__FirstRow(wt)
        patt = self.__NextPattern([this])
        while len(patt[-1]) < 2 * len(wt):
            newpatt = []
            for x in patt:
                newpatt += self.__NextPattern(x)
            patt = copy.deepcopy(newpatt)
        def revall(lst):
            return [(list(-sympy.Matrix(x)))[::-1] for x in lst]
        res = list(map(revall, patt))
        res.reverse()
        res.sort()
        return list(map(lambda x:x[0:-1], res))
    
    def __hsh(self, ptt):
        lst = [elt for q in ptt for elt in q]
        return sha1(str(lst).encode()).hexdigest()
    
    def __prodlst(self, lst):
        if lst != []:
            prd = functools.reduce(lambda x,y:x*y, lst)
        else:
            prd = 1
        return prd
    
    def J1(self, p):
        allpatt = self.patterns
 
        allpos = dict(zip([self.__hsh(patt.pattern) for patt in allpatt], range(0,len(allpatt))))

        
        def legitpatt(ptt):
            return (ptt in self.Gelfand_Tsetlin_patterns)
        def matrixentry_plus(j, ptt):
            num1 = self.__prodlst([(ptt.get_l(2*p-2,r)-ptt.get_l(2*p-1,j)-1)*(ptt.get_l(2*p-2,r)+ptt.get_l(2*p-1,j)) for r in range(1,p)])
            num2 = self.__prodlst([(ptt.get_l(2*p,r)-ptt.get_l(2*p-1,j)-1)*(ptt.get_l(2*p,r)+ptt.get_l(2*p-1,j)) for r in range(1,p+1)])
            denom = self.__prodlst([(ptt.get_l(2*p-1,r)**2 - ptt.get_l(2*p-1,j)**2)*(ptt.get_l(2*p-1,r)**2 - (ptt.get_l(2*p-1,j)+1)**2) for r in range(1,p+1) if r != j])
            return sympy.sqrt(sympy.Abs(sympy.Rational(num1 * num2, denom)))
        def matrixentry_minus(j, ptt):
            newptt = GTPattern(ptt.lowered(2*p-1,j))
            num1 = self.__prodlst([(newptt.get_l(2*p-2,r)-newptt.get_l(2*p-1,j)-1)*(newptt.get_l(2*p-2,r)+newptt.get_l(2*p-1,j)) for r in range(1,p)])
            num2 = self.__prodlst([(newptt.get_l(2*p,r)-newptt.get_l(2*p-1,j)-1)*(newptt.get_l(2*p,r)+newptt.get_l(2*p-1,j)) for r in range(1,p+1)])
            denom = self.__prodlst([(newptt.get_l(2*p-1,r)**2 - newptt.get_l(2*p-1,j)**2)*(newptt.get_l(2*p-1,r)**2 - (newptt.get_l(2*p-1,j)+1)**2) for r in range(1,p+1) if r != j])
            return -sympy.sqrt(sympy.Abs(sympy.Rational(num1 * num2, denom)))#
        dictrules1 = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.raised(2*p-1,j))]) for j in range(1, p+1) for ptt in allpatt if legitpatt(ptt.raised(2*p-1,j))], [matrixentry_plus(j, ptt) for j in range(1, p+1) for ptt in allpatt if legitpatt(ptt.raised(2*p-1,j))]))
        dictrules2 = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.lowered(2*p-1,j))]) for j in range(1, p+1) for ptt in allpatt if legitpatt(ptt.lowered(2*p-1,j))], [matrixentry_minus(j, ptt) for j in range(1, p+1) for ptt in allpatt if legitpatt(ptt.lowered(2*p-1,j))]))

        dictrules1.update(dictrules2)

        return sympy.matrices.SparseMatrix(len(allpatt),len(allpatt),dictrules1)
    
    
    def J2(self, p):
        allpatt = self.patterns
 
        allpos = dict(zip([self.__hsh(patt.pattern) for patt in allpatt], range(0,len(allpatt))))

        
        def legitpatt(ptt):
            return (ptt in self.Gelfand_Tsetlin_patterns)
        def matrixentry_plus(j, ptt):
            num1 = self.__prodlst([(ptt.get_l(2*p-1,r)**2-ptt.get_l(2*p,j)**2) for r in range(1,p+1)])
            num2 = self.__prodlst([(ptt.get_l(2*p+1,r)**2-ptt.get_l(2*p,j)**2) for r in range(1,p+2)])
            denom = (ptt.get_l(2*p,j)**2)*(4*ptt.get_l(2*p,j)**2 - 1) * self.__prodlst([(ptt.get_l(2*p,r)**2 - ptt.get_l(2*p,j)**2)*((ptt.get_l(2*p,r)-1)**2 - (ptt.get_l(2*p,j))**2) for r in range(1,p+1) if r != j])
            return sympy.sqrt(sympy.Abs(sympy.Rational(num1 * num2, denom)))
        def matrixentry_minus(j, ptt):
            newptt = GTPattern(ptt.lowered(2*p,j))
            num1 = self.__prodlst([(newptt.get_l(2*p-1,r)**2-newptt.get_l(2*p,j)**2) for r in range(1,p+1)])
            num2 = self.__prodlst([(newptt.get_l(2*p+1,r)**2-newptt.get_l(2*p,j)**2) for r in range(1,p+2)])
            denom = (newptt.get_l(2*p,j)**2)*(4*newptt.get_l(2*p,j)**2 - 1) * self.__prodlst([(newptt.get_l(2*p,r)**2 - newptt.get_l(2*p,j)**2)*((newptt.get_l(2*p,r)-1)**2 - (newptt.get_l(2*p,j))**2) for r in range(1,p+1) if r != j])
            return -sympy.sqrt(sympy.Abs(sympy.Rational(num1 * num2, denom)))
        def matrixentry_center(ptt):
            tst1 = self.__prodlst([ptt.get_l(2*p,r) * (ptt.get_l(2*p,r) - 1) for r in range(1,p+1)])
            if tst1 != 0:
                num = self.__prodlst([ptt.get_l(2*p-1,r) for r in range(1,p+1)]) * self.__prodlst([ptt.get_l(2*p+1,r) for r in range(1,p+2)])
                denom = self.__prodlst([ptt.get_l(2*p,r) * (ptt.get_l(2*p,r) - 1) for r in range(1,p+1)])
                return sympy.Rational(num, denom)
            else:
                if self.__prodlst([ptt.get_l(2*p-1,r) for r in range(1,p+1)]) * self.__prodlst([ptt.get_l(2*p+1,r) for r in range(1,p+2)]) == 0:
                    return 0
            
        dictrules1 = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.raised(2*p,j))]) for j in range(1, p+1) for ptt in allpatt if legitpatt(ptt.raised(2*p,j))], [matrixentry_plus(j, ptt) for j in range(1, p+1) for ptt in allpatt if legitpatt(ptt.raised(2*p,j))]))
        dictrules2 = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.lowered(2*p,j))]) for j in range(1, p+1) for ptt in allpatt if legitpatt(ptt.lowered(2*p,j))], [matrixentry_minus(j, ptt) for j in range(1, p+1) for ptt in allpatt if legitpatt(ptt.lowered(2*p,j))]))
        dictrules3 = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.pattern)]) for ptt in allpatt], [sympy.I * matrixentry_center(ptt) for ptt in allpatt]))

        dictrules1.update(dictrules2)

        dictrules1.update(dictrules3)

        return sympy.matrices.SparseMatrix(len(allpatt),len(allpatt),dictrules1)
    
    
    
    
    def J1N(self, p):
        allpatt = self.patterns
 
        allpos = dict(zip([self.__hsh(patt.pattern) for patt in allpatt], range(0,len(allpatt))))

        
        def legitpatt(ptt):
            return (ptt in self.Gelfand_Tsetlin_patterns)
        def matrixentry_plus(j, ptt):
            num1 = self.__prodlst([(ptt.get_l(2*p-2,r)+ptt.get_l(2*p-1,j))*(ptt.get_l(2*p-2,r)-ptt.get_l(2*p-1,j)-1) for r in range(1,p)])
            num2 = self.__prodlst([(ptt.get_l(2*p,r)+ptt.get_l(2*p-1,j))*(ptt.get_l(2*p,r)-ptt.get_l(2*p-1,j)-1) for r in range(1,p+1)])
            denom = self.__prodlst([(ptt.get_l(2*p-1,r) + ptt.get_l(2*p-1,j))*(ptt.get_l(2*p-1,r) - (ptt.get_l(2*p-1,j))) for r in range(1,p+1) if r != j])
            return (sympy.Abs(sympy.Rational(num1 * num2, denom)))
        def matrixentry_minus(j, ptt):
            newptt = GTPattern(ptt.lowered(2*p-1,j))
            denom = self.__prodlst([(ptt.get_l(2*p-1,r) - ptt.get_l(2*p-1,j))*(ptt.get_l(2*p-1,r) + (ptt.get_l(2*p-1,j))) for r in range(1,p+1) if r != j])
            return -(sympy.Abs(sympy.Rational(1, denom)))#
        dictrules1 = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.raised(2*p-1,j))]) for j in range(1, p+1) for ptt in allpatt if legitpatt(ptt.raised(2*p-1,j))], [matrixentry_plus(j, ptt) for j in range(1, p+1) for ptt in allpatt if legitpatt(ptt.raised(2*p-1,j))]))
        dictrules2 = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.lowered(2*p-1,j))]) for j in range(1, p+1) for ptt in allpatt if legitpatt(ptt.lowered(2*p-1,j))], [matrixentry_minus(j, ptt) for j in range(1, p+1) for ptt in allpatt if legitpatt(ptt.lowered(2*p-1,j))]))

        dictrules1.update(dictrules2)

        return sympy.matrices.SparseMatrix(len(allpatt),len(allpatt),dictrules1)
    
    def J2N(self, p):
        allpatt = self.patterns
 
        allpos = dict(zip([self.__hsh(patt.pattern) for patt in allpatt], range(0,len(allpatt))))

        
        def legitpatt(ptt):
            return (ptt in self.Gelfand_Tsetlin_patterns)
        def matrixentry_plus(j, ptt):
            num1 = self.__prodlst([(ptt.get_l(2*p-1,r)+ptt.get_l(2*p,j)) for r in range(1,p+1)])
            num2 = self.__prodlst([(ptt.get_l(2*p+1,r)+ptt.get_l(2*p,j)) for r in range(1,p+2)])
            denom = (2 * ptt.get_l(2*p,j)**2)*(4*ptt.get_l(2*p,j)**2 - 1) * self.__prodlst([(ptt.get_l(2*p,r) - ptt.get_l(2*p,j))*((ptt.get_l(2*p,r)) + (ptt.get_l(2*p,j)) - 1) for r in range(1,p+1) if r != j])
            return (sympy.Abs(sympy.Rational(num1 * num2, denom)))
        def matrixentry_minus(j, ptt):
            newptt = GTPattern(ptt.lowered(2*p,j))
            num1 = self.__prodlst([(ptt.get_l(2*p-1,r)-ptt.get_l(2*p,j)+1) for r in range(1,p+1)])
            num2 = self.__prodlst([(ptt.get_l(2*p+1,r)-ptt.get_l(2*p,j)+1) for r in range(1,p+2)])
            denom = self.__prodlst([(ptt.get_l(2*p,r) + ptt.get_l(2*p,j) - 1)*((ptt.get_l(2*p,r)) - (ptt.get_l(2*p,j))) for r in range(1,p+1) if r != j])
            return -(sympy.Abs(2*sympy.Rational(num1 * num2, denom)))
        def matrixentry_center(ptt):
            tst1 = self.__prodlst([ptt.get_l(2*p,r) * (ptt.get_l(2*p,r) - 1) for r in range(1,p+1)])
            if tst1 != 0:
                num = self.__prodlst([ptt.get_l(2*p-1,r) for r in range(1,p+1)]) * self.__prodlst([ptt.get_l(2*p+1,r) for r in range(1,p+2)])
                denom = self.__prodlst([ptt.get_l(2*p,r) * (ptt.get_l(2*p,r) - 1) for r in range(1,p+1)])
                return sympy.Rational(num, denom)
            else:
                if self.__prodlst([ptt.get_l(2*p-1,r) for r in range(1,p+1)]) * self.__prodlst([ptt.get_l(2*p+1,r) for r in range(1,p+2)]) == 0:
                    return 0
            
        dictrules1 = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.raised(2*p,j))]) for j in range(1, p+1) for ptt in allpatt if legitpatt(ptt.raised(2*p,j))], [matrixentry_plus(j, ptt) for j in range(1, p+1) for ptt in allpatt if legitpatt(ptt.raised(2*p,j))]))
        dictrules2 = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.lowered(2*p,j))]) for j in range(1, p+1) for ptt in allpatt if legitpatt(ptt.lowered(2*p,j))], [matrixentry_minus(j, ptt) for j in range(1, p+1) for ptt in allpatt if legitpatt(ptt.lowered(2*p,j))]))
        dictrules3 = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.pattern)]) for ptt in allpatt], [sympy.I * matrixentry_center(ptt) for ptt in allpatt]))

        dictrules1.update(dictrules2)

        dictrules1.update(dictrules3)

        return sympy.matrices.SparseMatrix(len(allpatt),len(allpatt),dictrules1)
    
    def libr(self, mata, matb):
        return mata * matb - matb * mata

    def get_cartan(self, vec, normalization = "rational"):
        if normalization == "rational":
            return functools.reduce(lambda x,y: x+y,([vec[k] * self.J2N(k) for k in range(0, len(vec))]))
        elif normalization == "unitary":
            return functools.reduce(lambda x,y: x+y,([vec[k] * self.J2(k) for k in range(0, len(vec))]))
    
    def get_simp_rootvec(self, r, normalization = "rational", root_type = "A-B"):
        n = len(self.highest_weight)
        if normalization == "rational":
            F1 = self.J1N
            F2 = self.J2N
        elif normalization == "unitary":
            F1 = self.J1
            F2 = self.J2
        if root_type == "A":
            return (F1(n) - sympy.I * self.libr(F2(n-1),F1(n)))/2
        elif root_type == "-A+B":
            return (self.libr(F2(r), F1(r+1)) + sympy.I * self.libr(self.libr(F2(r), F1(r+1)), F2(r+1)) - sympy.I * F1(r+1) + self.libr(F1(r+1), F2(r+1)))/4
        elif root_type == "A-B":
            return (self.libr(F2(r), F1(r+1)) - sympy.I * self.libr(self.libr(F2(r), F1(r+1)), F2(r+1)) + sympy.I * F1(r+1) + self.libr(F1(r+1), F2(r+1)))/4
        elif root_type == "-A":
            return (F1(n) + sympy.I * self.libr(F2(n-1),F1(n)))/2
           
    def __get_type_and_r(self, simprt, elt):
        pos = simprt.index(elt) + 1
        ln = len(simprt)
        if elt == 1:
            if pos <= ln - 1:
                return {"type":"A-B", "r":pos-1}
            else:
                return {"type":"A", "r":ln-1}
        elif elt == -1:
            if pos <= ln - 1:
                return {"type":"-A+B", "r":pos-1}
            else:
                return {"type":"-A", "r":ln-1}
    
    def get_root_vector(self, rt, normalization):
        storage = self.root_vectors
        if tuple(rt) in storage.keys():
            return storage[tuple(rt)]
        def vecpos(pos, ln):
            res = [0]*ln
            res[pos] = 1
            return res

        if abs(sum(rt)) == 1:
            assoc = self.__get_type_and_r(rt, sum(rt))
            return self.get_simp_rootvec(assoc["r"], normalization, root_type = assoc["type"])
        else:
            if sum(rt) > 0:
                allrts = self.root_data.root_strings
                pos = 0
                while list(sympy.Matrix(rt) - sympy.Matrix(vecpos(pos, len(rt)))) not in allrts:
                    pos += 1
                val = sympy.Rational(1, self.root_data.get_epsilon(vecpos(pos,len(rt)), list(sympy.Matrix(rt) - sympy.Matrix(vecpos(pos, len(rt)))))* (self.root_data.get_p(vecpos(pos,len(rt)), list(sympy.Matrix(rt) - sympy.Matrix(vecpos(pos,len(rt)))))+1))*self.libr(self.get_root_vector(vecpos(pos,len(rt)), normalization), self.get_root_vector(list(sympy.Matrix(rt) - sympy.Matrix(vecpos(pos,len(rt)))), normalization))/sympy.I
            elif sum(rt) < 0:
                allrts = self.root_data.root_strings
                pos = 0
                while list(sympy.Matrix(rt) + sympy.Matrix(vecpos(pos, len(rt)))) not in allrts:
                    pos += 1
                val = sympy.Rational(1, self.root_data.get_epsilon(list(-sympy.Matrix(vecpos(pos,len(rt)))), list(sympy.Matrix(rt), sympy.Matrix(vecpos(pos,len(rt)))))* (self.root_data.get_p(list(-sympy.Matrix(vecpos(pos,len(rt)))), list(sympy.Matrix(rt) + sympy.Matrix(vecpos(pos,len(rt)))))+1))*self.libr(self.get_root_vector(list(-sympy.Matrix(vecpos(pos,len(rt)))), normalization), self.get_root_vector(list(sympy.Matrix(rt) + sympy.Matrix(vecpos(pos,len(rt)))), normalization))/sympy.I
        storage[tuple(rt)] = val
        return val
    
    def unitarization(self):
        allpatt = self.patterns
        allpos = self.pattern_order
        def legitpatt(ptt):
            return (ptt in self.Gelfand_Tsetlin_patterns)
        def entry_1(k, j, ptt):
            pt1 = self.__prodlst([sympy.Rational(ptt.get_l(2*k-1,r)**2 - ptt.get_l(2*k-1,j)**2, ptt.get_l(2*k-1,r)**2 - (ptt.get_l(2*k-1,j) + 1)**2 ) for r in range(1,k+1) if r != j])
            pt2 = self.__prodlst([(ptt.get_l(2*k-2,r) - ptt.get_l(2*k-1,j) - 1) * (ptt.get_l(2*k-2,r) + ptt.get_l(2*k-1,j)) for r in range(1,k)])
            pt3 = self.__prodlst([(ptt.get_l(2*k,r) - ptt.get_l(2*k-1,j) - 1) * (ptt.get_l(2*k,r) + ptt.get_l(2*k-1,j)) for r in range(1,k+1)])
            return sympy.sqrt(abs( pt1 * sympy.Rational(1, pt2 * pt3) ))
        def entry_2(k, j, ptt):
            pt1 = self.__prodlst([sympy.Rational(ptt.get_l(2*k-1,r)**2 - ptt.get_l(2*k-1,j)**2, ptt.get_l(2*k-1,r)**2 - (ptt.get_l(2*k-1,j) - 1)**2 ) for r in range(1,k+1) if r != j])
            pt2 = self.__prodlst([(ptt.get_l(2*k-2,r) - ptt.get_l(2*k-1,j)) * (ptt.get_l(2*k-2,r) + ptt.get_l(2*k-1,j) - 1) for r in range(1,k)])
            pt3 = self.__prodlst([(ptt.get_l(2*k,r) - ptt.get_l(2*k-1,j)) * (ptt.get_l(2*k,r) + ptt.get_l(2*k-1,j) - 1) for r in range(1,k+1)])
            return sympy.sqrt(abs( pt1 * pt2 * pt3 ))
        def entry_3(k, j, ptt):
            pt1 = self.__prodlst([sympy.Rational((ptt.get_l(2*k,r) - ptt.get_l(2*k,j)) * (ptt.get_l(2*k,r) + ptt.get_l(2*k,j) - 1), (ptt.get_l(2*k,r) + ptt.get_l(2*k,j))*(ptt.get_l(2*k,r)-ptt.get_l(2*k,j)-1)) for r in range(1,k+1) if r != j])
            pt2 = self.__prodlst([sympy.Rational(ptt.get_l(2*k-1,r) - ptt.get_l(2*k,j), ptt.get_l(2*k-1,r) + ptt.get_l(2*k,j)) for r in range(1,k+1)])
            pt3 = self.__prodlst([sympy.Rational(ptt.get_l(2*k+1,r) - ptt.get_l(2*k,j), ptt.get_l(2*k+1,r) + ptt.get_l(2*k,j)) for r in range(1,k+2)])
            return sympy.sqrt(abs( pt1 * pt2 * pt3 * 4 * (ptt.get_l(2*k, j)**2) * (4 * (ptt.get_l(2*k, j)**2) - 1)))
        def entry_4(k, j, ptt):
            pt1 = self.__prodlst([sympy.Rational((ptt.get_l(2*k,r) - ptt.get_l(2*k,j)) * (ptt.get_l(2*k,r) + ptt.get_l(2*k,j) - 1), (ptt.get_l(2*k,r) + ptt.get_l(2*k,j) - 2)*(ptt.get_l(2*k,r)-ptt.get_l(2*k,j)+1)) for r in range(1,k+1) if r != j])
            pt2 = self.__prodlst([sympy.Rational(ptt.get_l(2*k-1,r) + ptt.get_l(2*k,j) - 1, ptt.get_l(2*k-1,r) - ptt.get_l(2*k,j) + 1) for r in range(1,k+1)])
            pt3 = self.__prodlst([sympy.Rational(ptt.get_l(2*k+1,r) + ptt.get_l(2*k,j) - 1, ptt.get_l(2*k+1,r) - ptt.get_l(2*k,j) + 1) for r in range(1,k+2)])
            return sympy.sqrt(abs( pt1 * pt2 * pt3 * sympy.Rational(1, 4 * ((ptt.get_l(2*k, j)-1)**2) * (4 * ((ptt.get_l(2*k, j)-1)**2) - 1))))
        part1 = [(allpos[self.__hsh(ptt.pattern)], [allpos[self.__hsh(ptt.raised(2*k-1,j))] , entry_1(k, j, ptt)]) for k in range(1, len(self.highest_weight)+1) for j in range(1,k+1) for ptt in allpatt if legitpatt(ptt.raised(2*k-1,j))]
        part2 = [(allpos[self.__hsh(ptt.pattern)], [allpos[self.__hsh(ptt.lowered(2*k-1,j))] , entry_2(k, j, ptt)]) for k in range(1, len(self.highest_weight)+1) for j in range(1,k+1) for ptt in allpatt if legitpatt(ptt.lowered(2*k-1,j))]
        part3 = [(allpos[self.__hsh(ptt.pattern)], [allpos[self.__hsh(ptt.raised(2*k,j))] , entry_3(k, j, ptt)]) for k in range(0, len(self.highest_weight)) for j in range(1,k+1) for ptt in allpatt if legitpatt(ptt.raised(2*k,j))]
        part4 = [(allpos[self.__hsh(ptt.pattern)], [allpos[self.__hsh(ptt.lowered(2*k,j))] , entry_4(k, j, ptt)]) for k in range(0, len(self.highest_weight)) for j in range(1,k+1) for ptt in allpatt if legitpatt(ptt.lowered(2*k,j))]
        allrules = part1 + part2 + part3 + part4
        allrules.sort()
        grp = itertools.groupby(allrules, lambda x:x[0])
        
        hashtable = dict([(node, [x[1] for x in val]) for node, val in grp])

        ylist = [0 for k in range(0, len(allpatt))]
        ylist[0] = sympy.Rational(1,1)
        def bfs(node):
            nodelist = hashtable[node]
            for nd in nodelist:
                if ylist[nd[0]] == 0:
                    ylist[nd[0] ] = ylist[node] * nd[1]
                    bfs(nd[0])
        bfs(0)
        res = [1/p for p in ylist]
        if hasattr(self, "unitarization_matrix") != True:
            setattr(self, "unitarization_matrix", res)
        return res