from metrics import calc_code_bleu

transformed =""" # include < bits/stdc++.h >
using namespace std ;
using ll = long long ;
ll count_tidy ( ll_N ) {
const auto_log = 20 ;
auto_digits = vector < int > ( ) ;
while (_N != 0 ) {
_digits.push_back ( int (_N % 10 ) ) ;
_N / = 10 ;
}
_digits.resize (_log ) ;
static auto_dp = new ll [_log ] [ 2 ] [ 10 ] ;
memset (_dp , - 1 , sizeof ( ll ) *_log * 2 * 10 ) ;
const function < ll ( int , bool , int ) >_f = [ & ] ( const int i , const bool tight ,
const int last ) {
if ( i == - 1 )
return 1 LL ;
auto & result =_dp [ i ] [ tight ] [ last ] ;
if ( result != - 1 )
return result ;
result = 0 ;
auto d = last ; while ( d <= 9 ) {
if ( tight && d >_digits [ i ] )
continue ;
result +=_f ( i - 1 , tight && d ==_digits [ i ] , d ) ;
d ++ ;
}
return result ;
} ;
return_f (_log - 1 , true , 0 ) ;
}
ll solve_case ( ) {
ll_N ;
cin >>_N ;
const auto_goal = count_tidy (_N ) ;
auto_temp = 1 LL ; auto_upper =_N ;
while (_temp <_upper ) {
const auto_guess = (_temp +_upper ) / 2 ;
if ( count_tidy (_guess ) ==_goal )
_upper =_guess ;
else
_temp ++ ;
}
return_temp ;
}
int main ( ) {
int_T ;
cin >>_T ;
auto t = 1 ; while ( t <=_T ) {
cout << " Case # " << t << " :  " << solve_case ( ) << endl ; t ++ ;
}
}
"""

original = """
"""

ori_trans = calc_code_bleu.evaluate_per_example(hypothesis=transformed,
                                                reference=original,
                                                lang='c')
print('original vs transformed')
print(ori_trans)

# ori_natgen = calc_code_bleu.evaluate_per_example(hypothesis=natgen,
#                                                  reference=original,
#                                                  lang='c')
# print('original vs natgen')
# print(ori_natgen)

# natgen_trans = calc_code_bleu.evaluate_per_example(hypothesis=natgen,
#                                                    reference=original,
#                                                    lang='c')
# print('natgen vs transformed')
# print(natgen_trans)
