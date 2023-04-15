import numpy as np
import sympy as sympy
import itertools
import functools
import operator
import copy

class RootDatAn:
    __epsilon_list = {}
    
    def __init__(self, n):
        self.rank = n
        self.roots = self.__RootsAn()
        self.simple_roots = self.__RootsAnSimple()
        self.coroots = self.__CorootsAn()
        self.simple_coroots = self.__CorootsAnSimple()
        self.cartan_matrix = self.__CartanMatrixAn()
        self.fundamental_weights = self.__FundamentalWeights()
        self.root_strings = self.__RootsAnSimpleBasis()
        self.positive_root_strings = self.__RootsAnSimpleBasisPos()
        self.highest_root_string = (self.positive_root_strings)[-1]
        self.highest_root = sympy.Matrix(self.highest_root_string).T * sympy.Matrix(self.simple_roots)
        self.root_order = self.__RootsAnOrder()
        self.positive_root_order = self.__RootsAnPosOrder()
    
    def __VectorByEntry(self, lst, ln):
    #List format: [[entry1, val1],[entry2, val2]]
        res = sympy.zeros(1, ln)
        for dat in lst:
            res[dat[0]] = dat[1]
        return res
    
    
    def __RootsAn(self):
        n = self.rank
        res = []
        for i in range(0, n):
            for j in range(i + 1, n + 1):
                res.append(self.__VectorByEntry([[i, 1], [j, -1]], n + 1))
        return res
    
    def __RootsAnSimple(self):
        n = self.rank
        res = []
        for i in range(0, n):
                res.append(self.__VectorByEntry([[i, 1], [i + 1, -1]], n + 1))
        return res
    
    def __CorootsAn(self):
        return [(2*x/(x.dot(x))) for x in self.roots]
    
    def __CorootsAnSimple(self):
        return [(2*x/(x.dot(x))) for x in self.simple_roots]
    
    def __CorootsInnerProduct(self):
        return sympy.Matrix([[a.dot(b) for a in self.simple_coroots] for b in self.simple_coroots])
    
    def __CartanMatrixAn(self):
        return sympy.Matrix([[a.dot(b) for a in self.simple_roots] for b in self.simple_coroots])
    
    def __FundamentalWeights(self):
        crt = self.simple_coroots
        ipinv = sympy.Matrix(self.__CorootsInnerProduct()).inv()
        fw = ((sympy.Matrix(crt).T)*ipinv).T.tolist()
        return [sympy.Matrix([[el - vec[-1] for el in vec]]) for vec in fw]
    
    def __RootsAnSimpleBasisPos(self):
        n = self.rank
        pos = []
        for lvl in range(1,n+1):
            for strt in range(0,n+1-lvl):
                tmp = [0]*(n)
                for i in range(strt,strt+lvl):
                    tmp[i] = 1
                pos.append(tmp)
        return pos
    
    def __RootsAnSimpleBasis(self):
        n = self.rank
        pos = []
        for lvl in range(1,n+1):
            for strt in range(0,n+1-lvl):
                tmp = [0]*(n)
                for i in range(strt,strt+lvl):
                    tmp[i] = 1
                pos.append(tmp)
        neg = []
        for lvl in range(1,n+1):
            for strt in range(0,n+1-lvl):
                tmp = [0]*(n)
                for i in range(strt,strt+lvl):
                    tmp[i] = -1
                neg.append(tmp)
        return neg+pos
    
    def __RootsAnOrder(self):
        allroots = self.root_strings
        allroots.sort(key=sum)
        return dict(zip(map(tuple,allroots), range(0,len(allroots))))
    
    def __RootsAnPosOrder(self):
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
    
    def __GetExtraspecialAn(self, alpha, beta):
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
                    val = -self.get_epsilon(list(-sympy.Matrix(alpha)), list(-sympy.Matrix(beta)))

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
                            exsp = self.__GetExtraspecialAn(alpha, beta)
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
        if k in range(1,len(self.pattern[0])+1):
            if i in range(1,k+1):
                self.pattern[len(self.pattern[0]) - k][i-1] += 1
            else:
                pass
        else:
            pass
    
    def raised(self, k, i):
        res = copy.deepcopy(self.pattern)
        if k in range(1,len(res[0])+1):
            if i in range(1,k+1):
                res[len(res[0]) - k][i-1] += 1
            else:
                pass
        else:
            pass
        return res
        
    def lower_pattern(self, k, i):
        if k in range(1,len(self.pattern[0])+1):
            if i in range(1,k+1):
                self.pattern[len(self.pattern[0]) - k][i-1] -= 1
            else:
                pass
        else:
            pass
        
    def lowered(self, k, i):
        res = copy.deepcopy(self.pattern)
        if k in range(1,len(res[0])+1):
            if i in range(1,k+1):
                res[len(res[0]) - k][i-1] -= 1
            else:
                pass
        else:
            pass
        return res
    
    def get_lambda(self, k, i):
        if k in range(1,len(self.pattern[0])+1):
            if i in range(1,k+1):
                return self.pattern[len(self.pattern[0]) - k][i-1]
            return 0
        return 0

    def get_l(self, k, i):
        if k in range(1,len(self.pattern[0])+1):
            if i in range(1,k+1):
                return self.get_lambda(k, i) - i + 1
            return 0
        return 0
        
    
class RepAn:
    def __init__(self, hwt):
        self.highest_weight = hwt
        self.root_data = RootDatAn(len(hwt) - 1)
        self.Gelfand_Tsetlin_patterns = self.__GenerateAllPatterns(hwt)
        self.patterns = [GTPattern(x) for x in self.Gelfand_Tsetlin_patterns]
        self.root_vectors = {}
    
    def __FirstRow(self, rw):
        return rw
    
    def __NextRow(self, rw):
        lst = [list(range(rw[r+1], rw[r]+1)) for r in range(0, len(rw)-1)]
        return [list(x) for x in itertools.product(*lst)]
    
    def __NextPattern(self, patt):
        lstrow = patt[-1]
        nxtrows = self.__NextRow(lstrow)
        return [patt+[x] for x in nxtrows]
    
    def __GenerateAllPatterns(self, wt):
        patt = self.__NextPattern([wt])
        while len(patt[-1]) < len(wt):
            newpatt = []
            for x in patt:
                newpatt += self.__NextPattern(x)
            patt = newpatt
        return patt
    
    def __hsh(self, ptt):
        return hash(tuple([elt for q in ptt for elt in q]))
    
    def libr(self, mata, matb):
        return mata*matb - matb*mata
    
    def E0(self, k):
        allpatt = self.patterns
        allpos = dict(zip([self.__hsh(patt.pattern) for patt in allpatt], range(0,len(allpatt))))
        dictrules = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.pattern)]) for ptt in allpatt], [sum([ptt.get_lambda(k,r) for r in range(1,k+1)]) - sum([ptt.get_lambda(k-1,r) for r in range(1,k)]) for ptt in allpatt]))
        return sympy.matrices.SparseMatrix(len(allpatt),len(allpatt),dictrules)
    
    def E1(self, k):
        return -self.E0(k)+ self.E0(k+1)
    
    def E2(self, k):
        allpatt = self.patterns
        allpos = dict(zip([self.__hsh(patt.pattern) for patt in allpatt], range(0,len(allpatt))))
        def legitpatt(ptt):
            return (ptt in self.Gelfand_Tsetlin_patterns)
        def matrixentry(i, ptt):
            num = -functools.reduce(lambda x,y:x*y, [(ptt.get_l(k,i) - ptt.get_l(k + 1, r)) for r in range(1,k+2)])
            def denomfun(i, r, ptt):
                if ptt.get_l(k, i) - ptt.get_l(k, r) != 0:
                    return ptt.get_l(k, i) - ptt.get_l(k, r)
                else:
                    return 1
            denom = functools.reduce(lambda x,y:x*y, [denomfun(i, r, ptt) for r in range(1,k+1)])
            return sympy.Rational(num, denom)
        dictrules = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.raised(k,i))]) for i in range(1, k+1) for ptt in allpatt if legitpatt(ptt.raised(k,i))], [matrixentry(i, ptt) for i in range(1, k+1) for ptt in allpatt if legitpatt(ptt.raised(k,i))]))
        return sympy.matrices.SparseMatrix(len(allpatt),len(allpatt),dictrules)
    
    def E3(self, k):
        allpatt = self.patterns
        allpos = dict(zip([self.__hsh(patt.pattern) for patt in allpatt], range(0,len(allpatt))))
        def legitpatt(ptt):
            return (ptt in self.Gelfand_Tsetlin_patterns)
        def matrixentry(i, ptt):
            if k > 1:
                num = functools.reduce(lambda x,y:x*y, [(ptt.get_l(k,i) - ptt.get_l(k - 1, r)) for r in range(1,k)])
            else:
                num = 1
            def denomfun(i, r, ptt):
                if ptt.get_l(k, i) - ptt.get_l(k, r) != 0:
                    return ptt.get_l(k, i) - ptt.get_l(k, r)
                else:
                    return 1
            denom = functools.reduce(lambda x,y:x*y, [denomfun(i, r, ptt) for r in range(1,k+1)])
            return sympy.Rational(num, denom)
        dictrules = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.lowered(k,i))]) for i in range(1, k+1) for ptt in allpatt if legitpatt(ptt.lowered(k,i))], [matrixentry(i, ptt) for i in range(1, k+1) for ptt in allpatt if legitpatt(ptt.lowered(k,i))]))
        return sympy.matrices.SparseMatrix(len(allpatt),len(allpatt),dictrules)
    
    def E0_hermitian(self, k):
        allpatt = self.patterns
        allpos = dict(zip([self.__hsh(patt.pattern) for patt in allpatt], range(0,len(allpatt))))
        dictrules = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.pattern)]) for ptt in allpatt], [sum([ptt.get_lambda(k,r) for r in range(1,k+1)]) - sum(ptt.get_lambda(k-1,r) for r in range(1,k)) for ptt in allpatt]))
        return sympy.matrices.SparseMatrix(len(allpatt),len(allpatt),dictrules)
    
    def E1_hermitian(self, k):
        return -self.E0(k)+ self.E0(k+1)

    def E2_hermitian(self, k):
        allpatt = self.patterns
        allpos = dict(zip([self.__hsh(patt.pattern) for patt in allpatt], range(0,len(allpatt))))
        def legitpatt(ptt):
            return (ptt in self.Gelfand_Tsetlin_patterns)
        def matrixentry(i, ptt):
            num1 = functools.reduce(lambda x,y:x*y, [(ptt.get_l(k + 1, r) - ptt.get_l(k, i)) for r in range(1,k+2)])
            if k > 1:
                num2 = functools.reduce(lambda x,y:x*y, [(ptt.get_l(k - 1, r) - ptt.get_l(k, i) - 1) for r in range(1,k)])
            else:
                num2 = 1
            def denomfun(i, r, ptt):
                if i != r:
                    return (ptt.get_l(k, r) - ptt.get_l(k, i))*(ptt.get_l(k, r) - ptt.get_l(k, i) - 1)
                else:
                    return 1
            denom = functools.reduce(lambda x,y:x*y, [denomfun(i, r, ptt) for r in range(1,k+1)])
            return sympy.I * sympy.sqrt(sympy.Rational(num1 * num2, denom))
        dictrules = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.raised(k,i))]) for i in range(1, k+1) for ptt in allpatt if legitpatt(ptt.raised(k,i))], [matrixentry(i, ptt) for i in range(1, k+1) for ptt in allpatt if legitpatt(ptt.raised(k,i))]))
        return sympy.matrices.SparseMatrix(len(allpatt),len(allpatt),dictrules)
    
    def E3_hermitian(self, k):
        allpatt = self.patterns
        allpos = dict(zip([self.__hsh(patt.pattern) for patt in allpatt], range(0,len(allpatt))))
        def legitpatt(ptt):
            return (ptt in self.Gelfand_Tsetlin_patterns)
        def matrixentry(i, ptt):
            num1 = functools.reduce(lambda x,y:x*y, [(ptt.get_l(k + 1, r) - ptt.get_l(k, i) + 1) for r in range(1,k+2)])
            if k > 1:
                num2 = functools.reduce(lambda x,y:x*y, [(ptt.get_l(k - 1, r) - ptt.get_l(k, i)) for r in range(1,k)])
            else:
                num2 = 1
            def denomfun(i, r, ptt):
                if i != r:
                    return (ptt.get_l(k, r) - ptt.get_l(k, i) + 1)*(ptt.get_l(k, r) - ptt.get_l(k, i))
                else:
                    return 1
            denom = functools.reduce(lambda x,y:x*y, [denomfun(i, r, ptt) for r in range(1,k+1)])
            return sympy.I * sympy.sqrt(sympy.Rational(num1 * num2, denom))
        dictrules = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.lowered(k,i))]) for i in range(1, k+1) for ptt in allpatt if legitpatt(ptt.lowered(k,i))], [matrixentry(i, ptt) for i in range(1, k+1) for ptt in allpatt if legitpatt(ptt.lowered(k,i))]))
        return sympy.matrices.SparseMatrix(len(allpatt),len(allpatt),dictrules)
    
    def get_root_vector(self, rt, normalization = "rational"):
        if normalization == "rational":
            F2 = self.E2
            F3 = self.E3
        elif normalization == "unitary":
            F2 = self.E2_hermitian
            F3 = self.E3_hermitian
        
        storage = self.root_vectors
        if tuple(rt) in storage.keys():
            return storage[tuple(rt)]
        def vecpos(pos, ln):
            res = [0]*ln
            res[pos] = 1
            return res

        if sum(rt) == 1:
            val = F2(rt.index(1)+1)
        elif sum(rt) == -1:
            val = F3(rt.index(-1)+1)
        else:
            if sum(rt) > 0:
                allrts = self.root_data.root_strings
                pos = 0
                while list(sympy.Matrix(rt) - sympy.Matrix(vecpos(pos, len(rt)))) not in allrts:
                    pos += 1
                val = sympy.Rational(1, self.root_data.get_epsilon(vecpos(pos,len(rt)), list(sympy.Matrix(rt) - sympy.Matrix(vecpos(pos, len(rt)))))* (self.root_data.get_p(vecpos(pos,len(rt)), list(sympy.Matrix(rt) - sympy.Matrix(vecpos(pos,len(rt)))))+1))*self.libr(self.get_root_vector(vecpos(pos,len(rt))), self.get_root_vector(list(sympy.Matrix(rt) - sympy.Matrix(vecpos(pos,len(rt))))))
            elif sum(rt) < 0:
                allrts = self.root_data.root_strings
                pos = 0
                while list(sympy.Matrix(rt) + sympy.Matrix(vecpos(pos, len(rt)))) not in allrts:
                    pos += 1
                val = sympy.Rational(1, self.root_data.get_epsilon(list(-sympy.Matrix(vecpos(pos,len(rt)))), list(sympy.Matrix(rt), sympy.Matrix(vecpos(pos,len(rt)))))* (self.root_data.get_p(list(-sympy.Matrix(vecpos(pos,len(rt)))), list(sympy.Matrix(rt) + sympy.Matrix(vecpos(pos,len(rt)))))+1))*self.libr(self.get_root_vector(list(-sympy.Matrix(vecpos(pos,len(rt))))), self.get_root_vector(list(sympy.Matrix(rt) + sympy.Matrix(vecpos(pos,len(rt))))))
        storage[tuple(rt)] = val
        return val
        

