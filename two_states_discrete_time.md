# Discrete time with concave utility

## Two states case 

The set up follows the set up in Aguiar and Amador (19), but in discrete time
and with concave utility. The utility is denoted by $u$. 

The bond is a long-duration exponential bond with decay rate $1 - \delta$. 

The default cost takes on two values (high and low). At  the beginning of every period, there is a probability $\lambda$ that the default cost is low. This realization is i.i.d. We denote by $\overline V$, the value of default if the default cost is low; and by $\underline V$, the value of default if the default cost is high.


Conditional on no default today, the value to a government that enters the period with $b$ bonds is:
\[
    V(b) = u(c) + \beta \lambda \max\{V(b'), \overline V\} + 
    \beta (1 - \lambda) \max \{ V(b'), \underline V\} 
\]
subject to:
\[
    y + q(b') (b' - (1 - \delta) b) = c + (r + \delta) b
\]
We let $B(b)$ denote the debt policy function associated with this problem. 

We impose that when indifferent, the country repays. So, default is then given by:
\[
    D(b, V_D) = \begin{cases}
      0 &, \text{ if } V(b) \ge V_D \\ 
      1 &, \text{ otherwise }  
    \end{cases}
\]
The ex-ante probability of default is then 
\[
    D(b) = \lambda D(b, \overline V) + (1-\lambda) D(b, 
    \underline V)
\]

Conditional on no default today, and an amount $b$ of bonds issued, the price of a bond is:
\[
    q(b) = (1 - D(b)) \left[
      \frac{  (r + \delta) + (1 - \delta) q (B(b)) }
      { 1 + r}
    \right]
\]

A Markov equilibrium is defined as a value function $V$, a price function $q$, a debt policy function $B$, and a default function $D$, such that the above equations are all satisfied. 

## Some useful values

The price of the bond at a stationary point (that is, when $B(b) = b$) solves 
\[
    q(b_{ss}) = (1 - D(b_{ss})) \left[
      \frac{  (r + \delta) + (1 - \delta) q (b_{ss}) }
      {1  + r}
    \right]
\]

We can also compute the associated stationary values:

\[
    V_{ss} (b_{ss}) =  
       u\left(y - \big(r + \delta (1 -  q(b_{ss}))
       \big) b_{ss} 
       \right)  \\ 
       + \beta (1 - \lambda) V_{ss}(b_{ss}) + 
       \beta \lambda \max \{ V_{ss}(b_ss), \overline V\}
\]

As in the paper, two potential points to consider. 

The stationary point at the safe zone in the savings equilibrium. In this case, $q_{ss}(\underline b^S) = 1$ and $V(\underline b^S) = \overline V$: 
\[
   (1 - \beta) \overline V = u(y - r \underline b^S)
\]
and thus 
\[
    \underline b ^S = \frac{y  - u^{-1} \left(
        ( 1 - \beta ) \overline V
    \right)}{r}
\]

The stationary debt level in the borrowing equilibrium, where $q = \underline q$ (below) and $V(\overline b^B) = \underline V$:
\[
    u\left( 
        y - (r + \delta ( 1 - \underline q)) \overline b^B
    \right) 
    = 
    (1 - \beta (1 - \lambda)) \underline V - \beta \lambda \overline V \\
    = (1 - \beta) \underline V - \beta \lambda (\overline V - 
    \underline V)  
\]
and thus
\[
     \overline b^B 
    = \frac{
    y - 
    u^{-1}
    \left(
        (1- \beta) \underline V - \beta \lambda (\overline V - 
        \underline V)
    \right)
    }
    {
     r + \delta (1 - \underline q)   
    }
\]

where $\underline q$ is:

\[
    \underline{q} = (1 - \lambda) \left[\frac{ (r + \delta) + (1 - \delta) \underline{q} }{ 1 + r}\right]
\]

which delivers
\[
    \underline{q} = \frac{r + \delta}{ \frac{r + \lambda}{1 - \lambda} + \delta}
\]