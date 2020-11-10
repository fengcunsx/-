# SVM

在这里再次写了一下SVM的推导，包含线性硬间隔和软间隔的的SVM，以及核方法实现的非线性的分类。

## SVM线性可分硬间隔

### 建模

我们的目的是使得到分类超平面$0=WX+B$的最小距离最大的$W$，具体的图片请翻阅之前写的SVM.里边有更加详细的介绍。

点到超平面的距离为：
$$
d=\frac{|wx+b|}{||W||_2}\\
我们在这里讨论二分类，分类结果为-1,+1,所以说上式可以化为:\\
d=\frac{y_i(wx_i+b)}{||W||_2}
$$
那么我们就可写出我们的目标优化函数：
$$
W=argmax_W\quad argmin_x{\frac{y_i(wx_i+b)}{\|W\|_2}}\\
argmax_W\quad \frac{1}{\|W\|_2}\quad argmin_x\quad y_i(Wx_i+b)
\\ 式1
$$
上式就是说到分类边界最近的点到分类边界的距离最大。

式1可以转化为：
$$
W=argmax_W\quad \frac{1}{\|W\|_2}\\
s.t. \quad y_i(wx_i-y_i)>\gamma
$$
不失一般性的我们把$\gamma$变为1，上式即：
$$
W=argmax_W\quad \frac{1}{\|W\|_2}\\
s.t. \quad y_i(wx_i+b)>1
$$
仅仅把$\gamma$乘到了左边。

变化一下：
$$
W=argmin_W\quad \frac{1}{2}\|W\|_2\\
s.t. \quad y_i(wx_i+b)>1
$$

**到这里我们就建模成功了，下面我们就是求解这个式子**

### 求解

我们这里将原问题转化为他的对偶问题，之后通过KKT条件来求解。

原问题：
$$
P=argmin_x\quad argmax_{\alpha}\quad \frac{1}{2}\|W\|_2+\sum_{i=1}^n\alpha_i\cdot [1-y_i(wx_i+b)]\\
s.t. \quad \alpha_i>0
$$
对偶问题：
$$
D=argmax_{\alpha}\quad argmin_x\quad  \frac{1}{2}\|W\|_2+\sum_{i=1}^n\alpha_i\cdot [1-y_i(wx_i+b)]\\
$$
满足KKT条件的时候：
$$
\nabla_{w.b}P=0\\
\alpha_i\geq 0\\
1-y_i(w\cdot x_i+b)\leq0\\
\alpha_i\cdot[1-y_i(w\cdot x_i+b)]=0
$$

对w进行求导：
$$
\nabla_wL=W-\sum_{i=1}^na_iy_ix_i=0\quad 即\\
W=\sum_{i=1}^na_iy_ix_i
$$
对b进行求导：
$$
\nabla_b=\sum_{i=1}^na_iy_i=0
$$
带回到L中对偶式子D变为：
$$
D=argmax_{\alpha}-\frac{1}{2}\sum_{i=1}^n\sum_{j=i}^n\alpha_i\alpha_jy_iy_jx_ix_j+\sum_{i=1}^n\alpha_i\\
s.t. \sum_{i=1}^n\alpha_iy_i=0
$$
即：
$$
D=argmin_{\alpha}\frac{1}{2}\sum_{i=1}^n\sum_{j=i}^n\alpha_i\alpha_jy_iy_jx_ix_j-\sum_{i=1}^n\alpha_i\\s.t. \sum_{i=1}^n\alpha_iy_i=0
$$
对$\alpha_i$求导得到：
$$
\frac{1}{2}\sum_{j=1}^n\alpha_jy_iy_jx_ix_j-1=0
$$


综上我们可以得到：
$$
W=\sum_{i=1}^na_iy_ix_i
$$
至于b，我们提供KKT条件：
$$
\alpha_i\cdot[1-y_i(w\cdot x_i+b)]=0
$$
提供$\alpha_i\ne0$的点$\alpha^*$可以得到：
$$
b=y_*-wx_*=y_*-\sum_{j=1}^n\alpha_j y_jx_jx_*
$$

## SVM线性可分软间隔

### 建模

我们允许存在一些误分类的点：

损失函数：
$$
L(x)=\frac{1}{2}\|w\|_2+C\sum_{i=1}^n\xi_i\\
s.t.\quad \xi_i\ge0\\
y_i(wx_i+b)\ge1-\xi_i
$$
我们这里的C是一个超参数，表示的是对误分类的容忍度。

当C大的时候，后一项会增大，后一项对L(x)增大贡献大，对误分类的容忍度减小。

当C小的时候，后一项会减小，后一项对L(x)增大贡献小，对误分类的容忍度增大

那么我们就可以写出**原问题**：
$$
P=argmin_{w,b,\xi}\quad argmax_{\alpha,\beta}\quad\frac{1}{2}\|w\|_2+C\sum_{i=1}^n\xi_i-\sum_{i=1}^n\alpha_i[{y_i(wx_i+b)-1+\xi_i}]-\sum_{i=1}^n\beta_i\xi_i\\
s.t.\quad \alpha_i\ge 0\\
\beta_i\ge0
$$
我们这里把最后两项写成负的是为了求导的时候好表示C

对偶问题：
$$
D=P=argmax_{\alpha,\beta}\quad argmax_{w,b,\xi}\quad\frac{1}{2}\|w\|_2+C\sum_{i=1}^n\xi_i-\sum_{i=1}^n\alpha_i[{y_i(wx_i+b)-1+\xi_i}]-\sum_{i=1}^n\beta_i\xi_i\\
$$
KKT条件：
$$
\nabla_{w.b,\xi}P=0\\
\alpha_i\geq 0\\
\beta-i\ge0\\
y_i(w\cdot x_i+b)-1+\xi_i\ge0\\
\alpha_i\cdot[y_i(w\cdot x_i+b)-1+\xi_i]=0\\
\beta_i\cdot \xi_i=0
$$
$\nabla_{w}P=0$可得：
$$
w=\sum_{i=1}^nx_Iy_i\alpha_i\\
式1
$$
$\nabla_{b}P=0$可得：
$$
\sum_{i=1}^n\alpha_iy_i=0\\式2
$$
$\nabla_{\xi_i}P=0$可得：
$$
C-\alpha_i-\beta_i=0\\式3
$$
将式1,2,3带入到对偶式子D里边去：
$$
D=min_{\alpha}\quad \frac12\sum_{i=1}^n\sum_{i=1}^n\alpha_i\alpha_bx_ix_jy_iy_j-\sum_{i=1}^n\alpha_i\\
s.t.\quad \sum_{i=1}^ny_i\alpha_i=0\\
C-\alpha_i-\beta_i=0
$$
综上我们可以得到：
$$
w=\sum_{i=1}^na_ix_iy_i
$$
对于b：我们提供KKT条件：
$$
\alpha_i\cdot[y_i(w\cdot x_i+b)-1+\xi_i]=0\\
\beta_i\cdot \xi_i=0\\
C=\alpha_i+\beta_i
$$
当$\alpha_i=0$的时候，$\beta=C,\beta\ne0$,所以说$\xi_i$=0，因此说：
$$
y_i(w\cdot x_i+b)=1
$$
即提供$\alpha_i\ne0$的点$\alpha^*$可以得到：
$$
b=y_*-wx_*=y_*-\sum_{j=1}^n\alpha_j y_jx_jx_*b=y_*-wx_*=y_*-\sum_{j=1}^n\alpha_j y_jx_jx_*
$$
