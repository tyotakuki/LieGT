import numpy as np
import sympy as sympy
import itertools
import functools
import operator
import copy

class RootDatCn:
    __epsilon_list = {}
    
    def __init__(self, n):
        self.rank = n
        self.roots = self.__RootsCn()
        self.simple_roots = self.__RootsCnSimple()
        self.coroots = self.__CorootsCn()
        self.simple_coroots = self.__CorootsCnSimple()
        self.cartan_matrix = self.__CartanMatrixCn()
        self.fundamental_weights = self.__FundamentalWeights()
        self.root_strings = self.__RootsCnSimpleBasis()
        self.positive_root_strings = [x for x in self.root_strings if sum(x) > 0]
        self.highest_root_string = (self.positive_root_strings)[-1]
        self.highest_root = sympy.Matrix(self.highest_root_string).T * sympy.Matrix(self.simple_roots)
        self.root_order = self.__RootsCnOrder()
        self.positive_root_order = self.__RootsCnPosOrder()

    
    def __VectorByEntry(self, lst, ln):
    #List format: [[entry1, val1],[entry2, val2]]
        res = sympy.zeros(1, ln)
        for dat in lst:
            res[dat[0]] = dat[1]
        return res
    
    def __RootsCn(self):
        n = self.rank
        res = []
        for j in range(0, n):
            res.append(self.__VectorByEntry([[j, 2]], n))
            res.append(self.__VectorByEntry([[j, -2]], n))
            for i in range(0, j):
                res.append(self.__VectorByEntry([[i, -1], [j, -1]], n))
                res.append(self.__VectorByEntry([[i, -1], [j, 1]], n))
                res.append(self.__VectorByEntry([[i, 1], [j, 1]], n))
                res.append(self.__VectorByEntry([[i, 1], [j, -1]], n))
        res.sort(key = tuple)
        return res
    
    def __RootsCnSimple(self):
        n = self.rank
        res = []
        for j in range(0, n-1):
            res.append(self.__VectorByEntry([[j, 1], [j+1, -1]], n))
        res.append(self.__VectorByEntry([[n-1, 2]], n))
        return res
    
    def __CorootsCn(self):
        return [(2*x/(x.dot(x))) for x in self.roots]
    
    def __CorootsCnSimple(self):
        return [(2*x/(x.dot(x))) for x in self.simple_roots]
    
    def __CorootsInnerProduct(self):
        return sympy.Matrix([[a.dot(b) for a in self.simple_coroots] for b in self.simple_coroots])
    
    def __CartanMatrixCn(self):
        return sympy.Matrix([[a.dot(b) for a in self.simple_roots] for b in self.simple_coroots])
    
    def __FundamentalWeights(self):
        crt = self.simple_coroots
        ipinv = sympy.Matrix(self.__CorootsInnerProduct()).inv()
        fw = ((sympy.Matrix(crt).T)*ipinv).T.tolist()
        return [sympy.Matrix([[el for el in vec]]) for vec in fw]
   
    def __RootsCnSimpleBasis(self):
        mat = (sympy.Matrix(self.roots) * sympy.Matrix(self.simple_roots).inv())
        res = [list(mat.row(k)) for k in range(0, mat.rows)]
        res.sort(key = sum)
        return res
    
    def __RootsCnOrder(self):
        allroots = self.root_strings
        allroots.sort(key=sum)
        return dict(zip(map(tuple,allroots), range(0,len(allroots))))
    
    def __RootsCnPosOrder(self):
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
    
    def __GetExtraspecialCn(self, alpha, beta):
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
                            exsp = self.__GetExtraspecialCn(alpha, beta)
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
                self.pattern[2 * (len(self.pattern[0]) - k)][i-1] += 1
            else:
                pass
        else:
            pass
    
    def raised(self, k, i):
        res = copy.deepcopy(self.pattern)
        if k in range(1,len(res[0])+1):
            if i in range(1,k+1):
                res[2 * (len(self.pattern[0]) - k)][i-1] += 1
            else:
                pass
        else:
            pass
        return res
    
    def raise_aux_pattern(self, k, i):
        if k in range(1,len(self.pattern[0])+1):
            if i in range(1,k+1):
                self.pattern[2 * (len(self.pattern[0]) - k) + 1][i-1] += 1
            else:
                pass
        else:
            pass
    
    def raised_aux(self, k, i):
        res = copy.deepcopy(self.pattern)
        if k in range(1,len(res[0])+1):
            if i in range(1,k+1):
                res[2 * (len(self.pattern[0]) - k) + 1][i-1] += 1
            else:
                pass
        else:
            pass
        return res
        
    def lower_pattern(self, k, i):
        if k in range(1,len(self.pattern[0])+1):
            if i in range(1,k+1):
                self.pattern[2 * (len(self.pattern[0]) - k)][i-1] -= 1
            else:
                pass
        else:
            pass
    
    def lowered(self, k, i):
        res = copy.deepcopy(self.pattern)
        if k in range(1,len(res[0])+1):
            if i in range(1,k+1):
                res[2 * (len(self.pattern[0]) - k)][i-1] -= 1
            else:
                pass
        else:
            pass
        return res
    
    def lower_aux_pattern(self, k, i):
        if k in range(1,len(self.pattern[0])+1):
            if i in range(1,k+1):
                self.pattern[2 * (len(self.pattern[0]) - k) + 1][i-1] -= 1
            else:
                pass
        else:
            pass
    
    def lowered_aux(self, k, i):
        res = copy.deepcopy(self.pattern)
        if k in range(1,len(res[0])+1):
            if i in range(1,k+1):
                res[2 * (len(self.pattern[0]) - k) + 1][i-1] -= 1
            else:
                pass
        else:
            pass
        return res
    
    def get_lambda(self, k, i):
        if k in range(1,len(self.pattern[0])+1):
            if i in range(1,k+1):
                return self.pattern[2* (len(self.pattern[0]) - k)][i-1]
            return 0
        return 0
    
    def get_lambda_aux(self, k, i):
        if k in range(1,len(self.pattern[0])+1):
            if i in range(1,k+1):
                return self.pattern[2* (len(self.pattern[0]) - k)+1][i-1]
            return 0
        return 0

    def get_l(self, k, i):
        if k in range(1,len(self.pattern[0])+1):
            if i in range(1,k+1):
                return self.get_lambda(k, i) - i
            return 0
        return 0
    
    def get_l_aux(self, k, i):
        if k in range(1,len(self.pattern[0])+1):
            if i in range(1,k+1):
                return self.get_lambda_aux(k, i) - i
            return 0
        return 0
    
    def sym_a(self, k, i):
        lst = [sympy.Rational(1,self.get_l_aux(k,a)-self.get_l_aux(k,i)) for a in range(1,k+1) if a != i]
        if lst != []:
            return functools.reduce(lambda x,y:x*y, lst)
        else:
            return 1
    
    def sym_b(self, k, i):

        fac1 = functools.reduce(lambda x,y:x*y, [self.get_l(k,a)-self.get_l_aux(k,i) for a in range(1,k+1)])
        if k > 1:
            fac2 = functools.reduce(lambda x,y:x*y, [self.get_l(k-1,a)-self.get_l_aux(k,i) for a in range(1,k)])
        else:
            fac2 = 1

        return 4 * self.sym_a(k, i) * self.get_l_aux(k, i) * fac1 * fac2
    
    def sym_c(self, k, i):
        lst = [sympy.Rational(1,self.get_l(k-1,i) ** 2 - self.get_l(k-1,a) ** 2) for a in range(1,k) if a != i]
        if lst != []:
            fac = functools.reduce(lambda x,y:x*y, lst)
        else:
            fac = 1
        return sympy.Rational(fac, 2 * self.get_l(k-1,i))
    
    def sym_d(self, k, i, j, m):
        lst1 = [(self.get_l(k-1,j)-self.get_l_aux(k,a))*(self.get_l(k-1,j)+self.get_l_aux(k,a)+1) for a in range(1,k+1) if a!=i]
        if lst1 != []:
            fac1 = functools.reduce(lambda x,y:x*y, lst1)
        else:
            fac1 = 1
        lst2 = [(self.get_l(k-1,j)-self.get_l_aux(k-1,a))*(self.get_l(k-1,j)+self.get_l_aux(k-1,a)+1) for a in range(1,k) if a!=m]
        if lst2 != []:
            fac2 = functools.reduce(lambda x,y:x*y, lst2)
        else:
            fac2 = 1

        return self.sym_a(k,i)*self.sym_a(k-1,m)*self.sym_c(k,j)*fac1*fac2
        
class RepCn:
    def __init__(self, hwt):
        self.highest_weight = hwt
        self.root_data = RootDatCn(len(hwt))
        self.Gelfand_Tsetlin_patterns = self.__GenerateAllPatterns(hwt)
        self.patterns = [GTPattern(x) for x in self.Gelfand_Tsetlin_patterns]
        self.root_vectors = {}
    
    def __FirstRow(self, rw):
        res = list(-sympy.Matrix(rw))
        res.reverse()
        return res
    
    def __NextAuxRow(self, rw):
        rw = [0] + rw
        lst = [list(range(rw[r+1], rw[r]+1)) for r in range(0, len(rw)-1)]
        return [list(x) for x in itertools.product(*lst)]
    
    def __NextRow(self, rw):
        lst = [list(range(rw[r+1], rw[r]+1)) for r in range(0, len(rw)-1)]
        return [list(x) for x in itertools.product(*lst)]
    
    def __NextCallback(self, patt, fun):
        lstrow = patt[-1]
        nxt = fun(lstrow)
        return [patt+[x] for x in nxt]
    
    def __NextPattern(self, patt):
        nxtaux = self.__NextCallback(patt, self.__NextRow)
        nxt = []
        for ptt in nxtaux:
            nxt = nxt + self.__NextCallback(ptt, self.__NextAuxRow)
        return nxt
    
    def __GenerateAllPatterns(self, wt):
        this = self.__FirstRow(wt)
        patt = self.__NextCallback([this], self.__NextAuxRow)
        while len(patt[-1]) < 2 * len(wt):
            newpatt = []
            for x in patt:
                newpatt += self.__NextPattern(x)
            patt = newpatt

        return patt
    
    def __hsh(self, ptt):
        return hash(tuple([elt for q in ptt for elt in q]))
    
    def libr(self, mata, matb):
        return mata*matb - matb*mata
    
    def E1(self, k):
        allpatt = self.patterns
        allpos = dict(zip([self.__hsh(patt.pattern) for patt in allpatt], range(0,len(allpatt))))
        dictrules = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.pattern)]) for ptt in allpatt], [-(2*sum([ptt.get_lambda_aux(k,i) for i in range(1,k+1)]) - sum([ptt.get_lambda(k,i) for i in range(1,k+1)]) - sum([ptt.get_lambda(k-1,i) for i in range(1,k)])) for ptt in allpatt]))
        return sympy.matrices.SparseMatrix(len(allpatt),len(allpatt),dictrules)
    
    def E2(self, k):
        allpatt = self.patterns
        allpos = dict(zip([self.__hsh(patt.pattern) for patt in allpatt], range(0,len(allpatt))))
        def legitpatt(ptt):
            return (ptt in self.Gelfand_Tsetlin_patterns)
        def matrixentry(i, ptt):
            num = ptt.sym_a(k,i)
            return sympy.Rational(num, 2)
        dictrules = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.raised_aux(k,i))]) for i in range(1, k+1) for ptt in allpatt if legitpatt(ptt.raised_aux(k,i))], [matrixentry(i, ptt) for i in range(1, k+1) for ptt in allpatt if legitpatt(ptt.raised_aux(k,i))]))
        return sympy.matrices.SparseMatrix(len(allpatt),len(allpatt),dictrules)
    
    def E3(self, k):
        allpatt = self.patterns
        allpos = dict(zip([self.__hsh(patt.pattern) for patt in allpatt], range(0,len(allpatt))))
        def legitpatt(ptt):
            return (ptt in self.Gelfand_Tsetlin_patterns)
        def matrixentry(i, ptt):
            num = ptt.sym_b(k,i)
            return sympy.Rational(num, 2)
        dictrules = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.lowered_aux(k,i))]) for i in range(1, k+1) for ptt in allpatt if legitpatt(ptt.lowered_aux(k,i))], [matrixentry(i, ptt) for i in range(1, k+1) for ptt in allpatt if legitpatt(ptt.lowered_aux(k,i))]))
        return sympy.matrices.SparseMatrix(len(allpatt),len(allpatt),dictrules)
    
    def __CombineTerms(self, dict1, dict2):
        for x in dict2.keys():
            if x in dict1:
                dict1[x] += dict2[x]
            else:
                dict1[x] = dict2[x]
        return dict1
    
    def E4(self, k):
        allpatt = self.patterns
        allpos = dict(zip([self.__hsh(patt.pattern) for patt in allpatt], range(0,len(allpatt))))
        def legitpatt(ptt):
            return (ptt in self.Gelfand_Tsetlin_patterns)
        def matrixentry1(i, ptt):

            num = sympy.I * ptt.sym_c(k,i)

            return num
        dictrules1 = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(ptt.lowered(k-1,i))]) for i in range(1, k) for ptt in allpatt if legitpatt(ptt.lowered(k-1,i))], [matrixentry1(i, ptt) for i in range(1, k) for ptt in allpatt if legitpatt(ptt.lowered(k-1,i))]))
        def matrixentry2(i,j,m,ptt):

            num = sympy.I * ptt.sym_d(k,i,j,m)

            return num
        dictrules2 = dict(zip([(allpos[self.__hsh(ptt.pattern)], allpos[self.__hsh(GTPattern(GTPattern(ptt.raised_aux(k,i)).raised(k-1,j)).raised_aux(k-1,m))]) for i in range(1, k+1) for j in range(1,k) for m in range(1,k) for ptt in allpatt if legitpatt(GTPattern(GTPattern(ptt.raised_aux(k,i)).raised(k-1,j)).raised_aux(k-1,m))], [matrixentry2(i,j,m, ptt) for i in range(1, k+1) for j in range(1,k) for m in range(1,k) for ptt in allpatt if legitpatt(GTPattern(GTPattern(ptt.raised_aux(k,i)).raised(k-1,j)).raised_aux(k-1,m))]))
        finalrules = self.__CombineTerms(dictrules1, dictrules2)

        return sympy.matrices.SparseMatrix(len(allpatt),len(allpatt),finalrules)
    
    def E5(self,k):
        return self.libr(self.E3(k-1), self.libr(self.E3(k), self.E4(k)))
    def E6(self,k):
        return self.libr(self.E4(k), self.E3(k))
    def E7(self,k):
        return self.libr(self.E5(k), self.E2(k))
    
    def get_root_vector(self, rt):
        storage = self.root_vectors
        if tuple(rt) in storage.keys():
            return storage[tuple(rt)]
        def vecpos(pos, ln):
            res = [0]*ln
            res[pos] = 1
            return res

        if sum(rt) == 1:
            if rt.index(1)+1 < len(rt):
                val = sympy.I * self.E6(rt.index(1)+2)
            else:
                val = self.E2(len(rt))
        
        elif sum(rt) == -1:
            if rt.index(-1)+1 < len(rt):
                val = sympy.I * self.E7(rt.index(-1)+2)
            else:
                val = self.E3(len(rt))
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
    
