??.
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.02unknown8??*
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
time_distributed/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nametime_distributed/kernel
?
+time_distributed/kernel/Read/ReadVariableOpReadVariableOptime_distributed/kernel*"
_output_shapes
:@*
dtype0
?
time_distributed/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nametime_distributed/bias
{
)time_distributed/bias/Read/ReadVariableOpReadVariableOptime_distributed/bias*
_output_shapes
:@*
dtype0
?
time_distributed_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ **
shared_nametime_distributed_1/kernel
?
-time_distributed_1/kernel/Read/ReadVariableOpReadVariableOptime_distributed_1/kernel*"
_output_shapes
:@ *
dtype0
?
time_distributed_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nametime_distributed_1/bias

+time_distributed_1/bias/Read/ReadVariableOpReadVariableOptime_distributed_1/bias*
_output_shapes
: *
dtype0
?
lstm_5/lstm_cell_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_namelstm_5/lstm_cell_7/kernel
?
-lstm_5/lstm_cell_7/kernel/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_7/kernel* 
_output_shapes
:
??*
dtype0
?
#lstm_5/lstm_cell_7/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*4
shared_name%#lstm_5/lstm_cell_7/recurrent_kernel
?
7lstm_5/lstm_cell_7/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_5/lstm_cell_7/recurrent_kernel*
_output_shapes
:	2?*
dtype0
?
lstm_5/lstm_cell_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namelstm_5/lstm_cell_7/bias
?
+lstm_5/lstm_cell_7/bias/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_7/bias*
_output_shapes	
:?*
dtype0
?
lstm_6/lstm_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2d**
shared_namelstm_6/lstm_cell_8/kernel
?
-lstm_6/lstm_cell_8/kernel/Read/ReadVariableOpReadVariableOplstm_6/lstm_cell_8/kernel*
_output_shapes

:2d*
dtype0
?
#lstm_6/lstm_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*4
shared_name%#lstm_6/lstm_cell_8/recurrent_kernel
?
7lstm_6/lstm_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_6/lstm_cell_8/recurrent_kernel*
_output_shapes

:d*
dtype0
?
lstm_6/lstm_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*(
shared_namelstm_6/lstm_cell_8/bias

+lstm_6/lstm_cell_8/bias/Read/ReadVariableOpReadVariableOplstm_6/lstm_cell_8/bias*
_output_shapes
:d*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_12/kernel/m
?
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_12/bias/m
y
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes
:*
dtype0
?
Adam/time_distributed/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/time_distributed/kernel/m
?
2Adam/time_distributed/kernel/m/Read/ReadVariableOpReadVariableOpAdam/time_distributed/kernel/m*"
_output_shapes
:@*
dtype0
?
Adam/time_distributed/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/time_distributed/bias/m
?
0Adam/time_distributed/bias/m/Read/ReadVariableOpReadVariableOpAdam/time_distributed/bias/m*
_output_shapes
:@*
dtype0
?
 Adam/time_distributed_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *1
shared_name" Adam/time_distributed_1/kernel/m
?
4Adam/time_distributed_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/time_distributed_1/kernel/m*"
_output_shapes
:@ *
dtype0
?
Adam/time_distributed_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/time_distributed_1/bias/m
?
2Adam/time_distributed_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/time_distributed_1/bias/m*
_output_shapes
: *
dtype0
?
 Adam/lstm_5/lstm_cell_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" Adam/lstm_5/lstm_cell_7/kernel/m
?
4Adam/lstm_5/lstm_cell_7/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_5/lstm_cell_7/kernel/m* 
_output_shapes
:
??*
dtype0
?
*Adam/lstm_5/lstm_cell_7/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*;
shared_name,*Adam/lstm_5/lstm_cell_7/recurrent_kernel/m
?
>Adam/lstm_5/lstm_cell_7/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_5/lstm_cell_7/recurrent_kernel/m*
_output_shapes
:	2?*
dtype0
?
Adam/lstm_5/lstm_cell_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_5/lstm_cell_7/bias/m
?
2Adam/lstm_5/lstm_cell_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_5/lstm_cell_7/bias/m*
_output_shapes	
:?*
dtype0
?
 Adam/lstm_6/lstm_cell_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2d*1
shared_name" Adam/lstm_6/lstm_cell_8/kernel/m
?
4Adam/lstm_6/lstm_cell_8/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_6/lstm_cell_8/kernel/m*
_output_shapes

:2d*
dtype0
?
*Adam/lstm_6/lstm_cell_8/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*;
shared_name,*Adam/lstm_6/lstm_cell_8/recurrent_kernel/m
?
>Adam/lstm_6/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_6/lstm_cell_8/recurrent_kernel/m*
_output_shapes

:d*
dtype0
?
Adam/lstm_6/lstm_cell_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*/
shared_name Adam/lstm_6/lstm_cell_8/bias/m
?
2Adam/lstm_6/lstm_cell_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_6/lstm_cell_8/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_12/kernel/v
?
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_12/bias/v
y
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes
:*
dtype0
?
Adam/time_distributed/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/time_distributed/kernel/v
?
2Adam/time_distributed/kernel/v/Read/ReadVariableOpReadVariableOpAdam/time_distributed/kernel/v*"
_output_shapes
:@*
dtype0
?
Adam/time_distributed/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/time_distributed/bias/v
?
0Adam/time_distributed/bias/v/Read/ReadVariableOpReadVariableOpAdam/time_distributed/bias/v*
_output_shapes
:@*
dtype0
?
 Adam/time_distributed_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *1
shared_name" Adam/time_distributed_1/kernel/v
?
4Adam/time_distributed_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/time_distributed_1/kernel/v*"
_output_shapes
:@ *
dtype0
?
Adam/time_distributed_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/time_distributed_1/bias/v
?
2Adam/time_distributed_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/time_distributed_1/bias/v*
_output_shapes
: *
dtype0
?
 Adam/lstm_5/lstm_cell_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" Adam/lstm_5/lstm_cell_7/kernel/v
?
4Adam/lstm_5/lstm_cell_7/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_5/lstm_cell_7/kernel/v* 
_output_shapes
:
??*
dtype0
?
*Adam/lstm_5/lstm_cell_7/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*;
shared_name,*Adam/lstm_5/lstm_cell_7/recurrent_kernel/v
?
>Adam/lstm_5/lstm_cell_7/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_5/lstm_cell_7/recurrent_kernel/v*
_output_shapes
:	2?*
dtype0
?
Adam/lstm_5/lstm_cell_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_5/lstm_cell_7/bias/v
?
2Adam/lstm_5/lstm_cell_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_5/lstm_cell_7/bias/v*
_output_shapes	
:?*
dtype0
?
 Adam/lstm_6/lstm_cell_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2d*1
shared_name" Adam/lstm_6/lstm_cell_8/kernel/v
?
4Adam/lstm_6/lstm_cell_8/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_6/lstm_cell_8/kernel/v*
_output_shapes

:2d*
dtype0
?
*Adam/lstm_6/lstm_cell_8/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*;
shared_name,*Adam/lstm_6/lstm_cell_8/recurrent_kernel/v
?
>Adam/lstm_6/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_6/lstm_cell_8/recurrent_kernel/v*
_output_shapes

:d*
dtype0
?
Adam/lstm_6/lstm_cell_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*/
shared_name Adam/lstm_6/lstm_cell_8/bias/v
?
2Adam/lstm_6/lstm_cell_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_6/lstm_cell_8/bias/v*
_output_shapes
:d*
dtype0

NoOpNoOp
?S
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?S
value?RB?R B?R
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
]
	layer
trainable_variables
regularization_losses
	variables
	keras_api
]
	layer
trainable_variables
regularization_losses
	variables
	keras_api
]
	layer
trainable_variables
regularization_losses
	variables
	keras_api
]
	layer
trainable_variables
regularization_losses
 	variables
!	keras_api
l
"cell
#
state_spec
$trainable_variables
%regularization_losses
&	variables
'	keras_api
l
(cell
)
state_spec
*trainable_variables
+regularization_losses
,	variables
-	keras_api
h

.kernel
/bias
0trainable_variables
1regularization_losses
2	variables
3	keras_api
?
4iter

5beta_1

6beta_2
	7decay
8learning_rate.m?/m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?.v?/v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?
V
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
.10
/11
 
V
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
.10
/11
?
Clayer_regularization_losses
Dnon_trainable_variables
	trainable_variables

regularization_losses
	variables
Elayer_metrics
Fmetrics

Glayers
 
h

9kernel
:bias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api

90
:1
 

90
:1
?
Llayer_regularization_losses
Mnon_trainable_variables
trainable_variables
regularization_losses
	variables
Nlayer_metrics
Ometrics

Players
h

;kernel
<bias
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api

;0
<1
 

;0
<1
?
Ulayer_regularization_losses
Vnon_trainable_variables
trainable_variables
regularization_losses
	variables
Wlayer_metrics
Xmetrics

Ylayers
R
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
 
 
 
?
^layer_regularization_losses
_non_trainable_variables
trainable_variables
regularization_losses
	variables
`layer_metrics
ametrics

blayers
R
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
 
 
 
?
glayer_regularization_losses
hnon_trainable_variables
trainable_variables
regularization_losses
 	variables
ilayer_metrics
jmetrics

klayers
~

=kernel
>recurrent_kernel
?bias
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
 

=0
>1
?2
 

=0
>1
?2
?
player_regularization_losses
qnon_trainable_variables
$trainable_variables
%regularization_losses
&	variables
rlayer_metrics

sstates
tmetrics

ulayers
~

@kernel
Arecurrent_kernel
Bbias
vtrainable_variables
wregularization_losses
x	variables
y	keras_api
 

@0
A1
B2
 

@0
A1
B2
?
zlayer_regularization_losses
{non_trainable_variables
*trainable_variables
+regularization_losses
,	variables
|layer_metrics

}states
~metrics

layers
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
 

.0
/1
?
 ?layer_regularization_losses
?non_trainable_variables
0trainable_variables
1regularization_losses
2	variables
?layer_metrics
?metrics
?layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEtime_distributed/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEtime_distributed/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEtime_distributed_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEtime_distributed_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_5/lstm_cell_7/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_5/lstm_cell_7/recurrent_kernel0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_5/lstm_cell_7/bias0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_6/lstm_cell_8/kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_6/lstm_cell_8/recurrent_kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_6/lstm_cell_8/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

?0
?1
1
0
1
2
3
4
5
6

90
:1
 

90
:1
?
 ?layer_regularization_losses
?non_trainable_variables
Htrainable_variables
Iregularization_losses
J	variables
?layer_metrics
?metrics
?layers
 
 
 
 

0

;0
<1
 

;0
<1
?
 ?layer_regularization_losses
?non_trainable_variables
Qtrainable_variables
Rregularization_losses
S	variables
?layer_metrics
?metrics
?layers
 
 
 
 

0
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
Ztrainable_variables
[regularization_losses
\	variables
?layer_metrics
?metrics
?layers
 
 
 
 

0
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
ctrainable_variables
dregularization_losses
e	variables
?layer_metrics
?metrics
?layers
 
 
 
 

0

=0
>1
?2
 

=0
>1
?2
?
 ?layer_regularization_losses
?non_trainable_variables
ltrainable_variables
mregularization_losses
n	variables
?layer_metrics
?metrics
?layers
 
 
 
 
 

"0

@0
A1
B2
 

@0
A1
B2
?
 ?layer_regularization_losses
?non_trainable_variables
vtrainable_variables
wregularization_losses
x	variables
?layer_metrics
?metrics
?layers
 
 
 
 
 

(0
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
~|
VARIABLE_VALUEAdam/dense_12/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/time_distributed/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/time_distributed/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/time_distributed_1/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/time_distributed_1/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_5/lstm_cell_7/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_5/lstm_cell_7/recurrent_kernel/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/lstm_5/lstm_cell_7/bias/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_6/lstm_cell_8/kernel/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_6/lstm_cell_8/recurrent_kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/lstm_6/lstm_cell_8/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/time_distributed/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/time_distributed/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/time_distributed_1/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/time_distributed_1/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_5/lstm_cell_7/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_5/lstm_cell_7/recurrent_kernel/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/lstm_5/lstm_cell_7/bias/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_6/lstm_cell_8/kernel/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_6/lstm_cell_8/recurrent_kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/lstm_6/lstm_cell_8/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
&serving_default_time_distributed_inputPlaceholder*8
_output_shapes&
$:"??????????????????
*
dtype0*-
shape$:"??????????????????

?
StatefulPartitionedCallStatefulPartitionedCall&serving_default_time_distributed_inputtime_distributed/kerneltime_distributed/biastime_distributed_1/kerneltime_distributed_1/biaslstm_5/lstm_cell_7/kernel#lstm_5/lstm_cell_7/recurrent_kernellstm_5/lstm_cell_7/biaslstm_6/lstm_cell_8/kernel#lstm_6/lstm_cell_8/recurrent_kernellstm_6/lstm_cell_8/biasdense_12/kerneldense_12/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_203239
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp+time_distributed/kernel/Read/ReadVariableOp)time_distributed/bias/Read/ReadVariableOp-time_distributed_1/kernel/Read/ReadVariableOp+time_distributed_1/bias/Read/ReadVariableOp-lstm_5/lstm_cell_7/kernel/Read/ReadVariableOp7lstm_5/lstm_cell_7/recurrent_kernel/Read/ReadVariableOp+lstm_5/lstm_cell_7/bias/Read/ReadVariableOp-lstm_6/lstm_cell_8/kernel/Read/ReadVariableOp7lstm_6/lstm_cell_8/recurrent_kernel/Read/ReadVariableOp+lstm_6/lstm_cell_8/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp2Adam/time_distributed/kernel/m/Read/ReadVariableOp0Adam/time_distributed/bias/m/Read/ReadVariableOp4Adam/time_distributed_1/kernel/m/Read/ReadVariableOp2Adam/time_distributed_1/bias/m/Read/ReadVariableOp4Adam/lstm_5/lstm_cell_7/kernel/m/Read/ReadVariableOp>Adam/lstm_5/lstm_cell_7/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_5/lstm_cell_7/bias/m/Read/ReadVariableOp4Adam/lstm_6/lstm_cell_8/kernel/m/Read/ReadVariableOp>Adam/lstm_6/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_6/lstm_cell_8/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp2Adam/time_distributed/kernel/v/Read/ReadVariableOp0Adam/time_distributed/bias/v/Read/ReadVariableOp4Adam/time_distributed_1/kernel/v/Read/ReadVariableOp2Adam/time_distributed_1/bias/v/Read/ReadVariableOp4Adam/lstm_5/lstm_cell_7/kernel/v/Read/ReadVariableOp>Adam/lstm_5/lstm_cell_7/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_5/lstm_cell_7/bias/v/Read/ReadVariableOp4Adam/lstm_6/lstm_cell_8/kernel/v/Read/ReadVariableOp>Adam/lstm_6/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_6/lstm_cell_8/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_206075
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_12/kerneldense_12/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetime_distributed/kerneltime_distributed/biastime_distributed_1/kerneltime_distributed_1/biaslstm_5/lstm_cell_7/kernel#lstm_5/lstm_cell_7/recurrent_kernellstm_5/lstm_cell_7/biaslstm_6/lstm_cell_8/kernel#lstm_6/lstm_cell_8/recurrent_kernellstm_6/lstm_cell_8/biastotalcounttotal_1count_1Adam/dense_12/kernel/mAdam/dense_12/bias/mAdam/time_distributed/kernel/mAdam/time_distributed/bias/m Adam/time_distributed_1/kernel/mAdam/time_distributed_1/bias/m Adam/lstm_5/lstm_cell_7/kernel/m*Adam/lstm_5/lstm_cell_7/recurrent_kernel/mAdam/lstm_5/lstm_cell_7/bias/m Adam/lstm_6/lstm_cell_8/kernel/m*Adam/lstm_6/lstm_cell_8/recurrent_kernel/mAdam/lstm_6/lstm_cell_8/bias/mAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/time_distributed/kernel/vAdam/time_distributed/bias/v Adam/time_distributed_1/kernel/vAdam/time_distributed_1/bias/v Adam/lstm_5/lstm_cell_7/kernel/v*Adam/lstm_5/lstm_cell_7/recurrent_kernel/vAdam/lstm_5/lstm_cell_7/bias/v Adam/lstm_6/lstm_cell_8/kernel/v*Adam/lstm_6/lstm_cell_8/recurrent_kernel/vAdam/lstm_6/lstm_cell_8/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_206220??(
?W
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_204959

inputs.
*lstm_cell_7_matmul_readvariableop_resource0
,lstm_cell_7_matmul_1_readvariableop_resource/
+lstm_cell_7_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!lstm_cell_7/MatMul/ReadVariableOp?
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/MatMul?
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_7/MatMul_1/ReadVariableOp?
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/MatMul_1?
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/add?
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_7/BiasAdd/ReadVariableOp?
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/BiasAddh
lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/Const|
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/split/split_dim?
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_7/split?
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid?
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid_1?
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mulz
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Relu?
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mul_1?
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/add_1?
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid_2y
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Relu_1?
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_204874*
condR
while_cond_204873*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?/
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_203101

inputs
time_distributed_203061
time_distributed_203063
time_distributed_1_203068
time_distributed_1_203070
lstm_5_203081
lstm_5_203083
lstm_5_203085
lstm_6_203088
lstm_6_203090
lstm_6_203092
dense_12_203095
dense_12_203097
identity?? dense_12/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?lstm_6/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_203061time_distributed_203063*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"??????????????????	@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_2006632*
(time_distributed/StatefulPartitionedCall?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????
      2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????
2
time_distributed/Reshape?
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_203068time_distributed_1_203070*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"?????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_2007942,
*time_distributed_1/StatefulPartitionedCall?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????	   @   2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????	@2
time_distributed_1/Reshape?
"time_distributed_2/PartitionedCallPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_2009012$
"time_distributed_2/PartitionedCall?
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2"
 time_distributed_2/Reshape/shape?
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_2/Reshape?
"time_distributed_3/PartitionedCallPartitionedCall+time_distributed_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_2009912$
"time_distributed_3/PartitionedCall?
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2"
 time_distributed_3/Reshape/shape?
time_distributed_3/ReshapeReshape+time_distributed_2/PartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_3/Reshape?
lstm_5/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0lstm_5_203081lstm_5_203083lstm_5_203085*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_2024672 
lstm_5/StatefulPartitionedCall?
lstm_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0lstm_6_203088lstm_6_203090lstm_6_203092*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_6_layer_call_and_return_conditional_losses_2028022 
lstm_6/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall'lstm_6/StatefulPartitionedCall:output:0dense_12_203095dense_12_203097*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_2029952"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall^lstm_6/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall2@
lstm_6/StatefulPartitionedCalllstm_6/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:` \
8
_output_shapes&
$:"??????????????????

 
_user_specified_nameinputs
?
?
while_cond_204545
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_204545___redundant_placeholder04
0while_while_cond_204545___redundant_placeholder14
0while_while_cond_204545___redundant_placeholder24
0while_while_cond_204545___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_205376
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_205376___redundant_placeholder04
0while_while_cond_205376___redundant_placeholder14
0while_while_cond_205376___redundant_placeholder24
0while_while_cond_205376___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_200943

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
ę
?
!__inference__wrapped_model_200569
time_distributed_inputV
Rsequential_5_time_distributed_conv1d_2_conv1d_expanddims_1_readvariableop_resourceJ
Fsequential_5_time_distributed_conv1d_2_biasadd_readvariableop_resourceX
Tsequential_5_time_distributed_1_conv1d_3_conv1d_expanddims_1_readvariableop_resourceL
Hsequential_5_time_distributed_1_conv1d_3_biasadd_readvariableop_resourceB
>sequential_5_lstm_5_lstm_cell_7_matmul_readvariableop_resourceD
@sequential_5_lstm_5_lstm_cell_7_matmul_1_readvariableop_resourceC
?sequential_5_lstm_5_lstm_cell_7_biasadd_readvariableop_resourceB
>sequential_5_lstm_6_lstm_cell_8_matmul_readvariableop_resourceD
@sequential_5_lstm_6_lstm_cell_8_matmul_1_readvariableop_resourceC
?sequential_5_lstm_6_lstm_cell_8_biasadd_readvariableop_resource8
4sequential_5_dense_12_matmul_readvariableop_resource9
5sequential_5_dense_12_biasadd_readvariableop_resource
identity??sequential_5/lstm_5/while?sequential_5/lstm_6/while?
#sequential_5/time_distributed/ShapeShapetime_distributed_input*
T0*
_output_shapes
:2%
#sequential_5/time_distributed/Shape?
1sequential_5/time_distributed/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1sequential_5/time_distributed/strided_slice/stack?
3sequential_5/time_distributed/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_5/time_distributed/strided_slice/stack_1?
3sequential_5/time_distributed/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_5/time_distributed/strided_slice/stack_2?
+sequential_5/time_distributed/strided_sliceStridedSlice,sequential_5/time_distributed/Shape:output:0:sequential_5/time_distributed/strided_slice/stack:output:0<sequential_5/time_distributed/strided_slice/stack_1:output:0<sequential_5/time_distributed/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_5/time_distributed/strided_slice?
+sequential_5/time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????
      2-
+sequential_5/time_distributed/Reshape/shape?
%sequential_5/time_distributed/ReshapeReshapetime_distributed_input4sequential_5/time_distributed/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????
2'
%sequential_5/time_distributed/Reshape?
<sequential_5/time_distributed/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2>
<sequential_5/time_distributed/conv1d_2/conv1d/ExpandDims/dim?
8sequential_5/time_distributed/conv1d_2/conv1d/ExpandDims
ExpandDims.sequential_5/time_distributed/Reshape:output:0Esequential_5/time_distributed/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
2:
8sequential_5/time_distributed/conv1d_2/conv1d/ExpandDims?
Isequential_5/time_distributed/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpRsequential_5_time_distributed_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02K
Isequential_5/time_distributed/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
>sequential_5/time_distributed/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2@
>sequential_5/time_distributed/conv1d_2/conv1d/ExpandDims_1/dim?
:sequential_5/time_distributed/conv1d_2/conv1d/ExpandDims_1
ExpandDimsQsequential_5/time_distributed/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0Gsequential_5/time_distributed/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2<
:sequential_5/time_distributed/conv1d_2/conv1d/ExpandDims_1?
-sequential_5/time_distributed/conv1d_2/conv1dConv2DAsequential_5/time_distributed/conv1d_2/conv1d/ExpandDims:output:0Csequential_5/time_distributed/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????	@*
paddingVALID*
strides
2/
-sequential_5/time_distributed/conv1d_2/conv1d?
5sequential_5/time_distributed/conv1d_2/conv1d/SqueezeSqueeze6sequential_5/time_distributed/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????	@*
squeeze_dims

?????????27
5sequential_5/time_distributed/conv1d_2/conv1d/Squeeze?
=sequential_5/time_distributed/conv1d_2/BiasAdd/ReadVariableOpReadVariableOpFsequential_5_time_distributed_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02?
=sequential_5/time_distributed/conv1d_2/BiasAdd/ReadVariableOp?
.sequential_5/time_distributed/conv1d_2/BiasAddBiasAdd>sequential_5/time_distributed/conv1d_2/conv1d/Squeeze:output:0Esequential_5/time_distributed/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????	@20
.sequential_5/time_distributed/conv1d_2/BiasAdd?
+sequential_5/time_distributed/conv1d_2/ReluRelu7sequential_5/time_distributed/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????	@2-
+sequential_5/time_distributed/conv1d_2/Relu?
/sequential_5/time_distributed/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????21
/sequential_5/time_distributed/Reshape_1/shape/0?
/sequential_5/time_distributed/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	21
/sequential_5/time_distributed/Reshape_1/shape/2?
/sequential_5/time_distributed/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@21
/sequential_5/time_distributed/Reshape_1/shape/3?
-sequential_5/time_distributed/Reshape_1/shapePack8sequential_5/time_distributed/Reshape_1/shape/0:output:04sequential_5/time_distributed/strided_slice:output:08sequential_5/time_distributed/Reshape_1/shape/2:output:08sequential_5/time_distributed/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2/
-sequential_5/time_distributed/Reshape_1/shape?
'sequential_5/time_distributed/Reshape_1Reshape9sequential_5/time_distributed/conv1d_2/Relu:activations:06sequential_5/time_distributed/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????	@2)
'sequential_5/time_distributed/Reshape_1?
-sequential_5/time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????
      2/
-sequential_5/time_distributed/Reshape_2/shape?
'sequential_5/time_distributed/Reshape_2Reshapetime_distributed_input6sequential_5/time_distributed/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????
2)
'sequential_5/time_distributed/Reshape_2?
%sequential_5/time_distributed_1/ShapeShape0sequential_5/time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:2'
%sequential_5/time_distributed_1/Shape?
3sequential_5/time_distributed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential_5/time_distributed_1/strided_slice/stack?
5sequential_5/time_distributed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/time_distributed_1/strided_slice/stack_1?
5sequential_5/time_distributed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/time_distributed_1/strided_slice/stack_2?
-sequential_5/time_distributed_1/strided_sliceStridedSlice.sequential_5/time_distributed_1/Shape:output:0<sequential_5/time_distributed_1/strided_slice/stack:output:0>sequential_5/time_distributed_1/strided_slice/stack_1:output:0>sequential_5/time_distributed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_5/time_distributed_1/strided_slice?
-sequential_5/time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????	   @   2/
-sequential_5/time_distributed_1/Reshape/shape?
'sequential_5/time_distributed_1/ReshapeReshape0sequential_5/time_distributed/Reshape_1:output:06sequential_5/time_distributed_1/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????	@2)
'sequential_5/time_distributed_1/Reshape?
>sequential_5/time_distributed_1/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2@
>sequential_5/time_distributed_1/conv1d_3/conv1d/ExpandDims/dim?
:sequential_5/time_distributed_1/conv1d_3/conv1d/ExpandDims
ExpandDims0sequential_5/time_distributed_1/Reshape:output:0Gsequential_5/time_distributed_1/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????	@2<
:sequential_5/time_distributed_1/conv1d_3/conv1d/ExpandDims?
Ksequential_5/time_distributed_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpTsequential_5_time_distributed_1_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02M
Ksequential_5/time_distributed_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
@sequential_5/time_distributed_1/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2B
@sequential_5/time_distributed_1/conv1d_3/conv1d/ExpandDims_1/dim?
<sequential_5/time_distributed_1/conv1d_3/conv1d/ExpandDims_1
ExpandDimsSsequential_5/time_distributed_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0Isequential_5/time_distributed_1/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2>
<sequential_5/time_distributed_1/conv1d_3/conv1d/ExpandDims_1?
/sequential_5/time_distributed_1/conv1d_3/conv1dConv2DCsequential_5/time_distributed_1/conv1d_3/conv1d/ExpandDims:output:0Esequential_5/time_distributed_1/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
21
/sequential_5/time_distributed_1/conv1d_3/conv1d?
7sequential_5/time_distributed_1/conv1d_3/conv1d/SqueezeSqueeze8sequential_5/time_distributed_1/conv1d_3/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????29
7sequential_5/time_distributed_1/conv1d_3/conv1d/Squeeze?
?sequential_5/time_distributed_1/conv1d_3/BiasAdd/ReadVariableOpReadVariableOpHsequential_5_time_distributed_1_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?sequential_5/time_distributed_1/conv1d_3/BiasAdd/ReadVariableOp?
0sequential_5/time_distributed_1/conv1d_3/BiasAddBiasAdd@sequential_5/time_distributed_1/conv1d_3/conv1d/Squeeze:output:0Gsequential_5/time_distributed_1/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 22
0sequential_5/time_distributed_1/conv1d_3/BiasAdd?
-sequential_5/time_distributed_1/conv1d_3/ReluRelu9sequential_5/time_distributed_1/conv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2/
-sequential_5/time_distributed_1/conv1d_3/Relu?
1sequential_5/time_distributed_1/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????23
1sequential_5/time_distributed_1/Reshape_1/shape/0?
1sequential_5/time_distributed_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :23
1sequential_5/time_distributed_1/Reshape_1/shape/2?
1sequential_5/time_distributed_1/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_5/time_distributed_1/Reshape_1/shape/3?
/sequential_5/time_distributed_1/Reshape_1/shapePack:sequential_5/time_distributed_1/Reshape_1/shape/0:output:06sequential_5/time_distributed_1/strided_slice:output:0:sequential_5/time_distributed_1/Reshape_1/shape/2:output:0:sequential_5/time_distributed_1/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:21
/sequential_5/time_distributed_1/Reshape_1/shape?
)sequential_5/time_distributed_1/Reshape_1Reshape;sequential_5/time_distributed_1/conv1d_3/Relu:activations:08sequential_5/time_distributed_1/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2+
)sequential_5/time_distributed_1/Reshape_1?
/sequential_5/time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????	   @   21
/sequential_5/time_distributed_1/Reshape_2/shape?
)sequential_5/time_distributed_1/Reshape_2Reshape0sequential_5/time_distributed/Reshape_1:output:08sequential_5/time_distributed_1/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????	@2+
)sequential_5/time_distributed_1/Reshape_2?
%sequential_5/time_distributed_2/ShapeShape2sequential_5/time_distributed_1/Reshape_1:output:0*
T0*
_output_shapes
:2'
%sequential_5/time_distributed_2/Shape?
3sequential_5/time_distributed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential_5/time_distributed_2/strided_slice/stack?
5sequential_5/time_distributed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/time_distributed_2/strided_slice/stack_1?
5sequential_5/time_distributed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/time_distributed_2/strided_slice/stack_2?
-sequential_5/time_distributed_2/strided_sliceStridedSlice.sequential_5/time_distributed_2/Shape:output:0<sequential_5/time_distributed_2/strided_slice/stack:output:0>sequential_5/time_distributed_2/strided_slice/stack_1:output:0>sequential_5/time_distributed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_5/time_distributed_2/strided_slice?
-sequential_5/time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2/
-sequential_5/time_distributed_2/Reshape/shape?
'sequential_5/time_distributed_2/ReshapeReshape2sequential_5/time_distributed_1/Reshape_1:output:06sequential_5/time_distributed_2/Reshape/shape:output:0*
T0*+
_output_shapes
:????????? 2)
'sequential_5/time_distributed_2/Reshape?
>sequential_5/time_distributed_2/max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_5/time_distributed_2/max_pooling1d_1/ExpandDims/dim?
:sequential_5/time_distributed_2/max_pooling1d_1/ExpandDims
ExpandDims0sequential_5/time_distributed_2/Reshape:output:0Gsequential_5/time_distributed_2/max_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2<
:sequential_5/time_distributed_2/max_pooling1d_1/ExpandDims?
7sequential_5/time_distributed_2/max_pooling1d_1/MaxPoolMaxPoolCsequential_5/time_distributed_2/max_pooling1d_1/ExpandDims:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
29
7sequential_5/time_distributed_2/max_pooling1d_1/MaxPool?
7sequential_5/time_distributed_2/max_pooling1d_1/SqueezeSqueeze@sequential_5/time_distributed_2/max_pooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims
29
7sequential_5/time_distributed_2/max_pooling1d_1/Squeeze?
1sequential_5/time_distributed_2/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????23
1sequential_5/time_distributed_2/Reshape_1/shape/0?
1sequential_5/time_distributed_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :23
1sequential_5/time_distributed_2/Reshape_1/shape/2?
1sequential_5/time_distributed_2/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_5/time_distributed_2/Reshape_1/shape/3?
/sequential_5/time_distributed_2/Reshape_1/shapePack:sequential_5/time_distributed_2/Reshape_1/shape/0:output:06sequential_5/time_distributed_2/strided_slice:output:0:sequential_5/time_distributed_2/Reshape_1/shape/2:output:0:sequential_5/time_distributed_2/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:21
/sequential_5/time_distributed_2/Reshape_1/shape?
)sequential_5/time_distributed_2/Reshape_1Reshape@sequential_5/time_distributed_2/max_pooling1d_1/Squeeze:output:08sequential_5/time_distributed_2/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2+
)sequential_5/time_distributed_2/Reshape_1?
/sequential_5/time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       21
/sequential_5/time_distributed_2/Reshape_2/shape?
)sequential_5/time_distributed_2/Reshape_2Reshape2sequential_5/time_distributed_1/Reshape_1:output:08sequential_5/time_distributed_2/Reshape_2/shape:output:0*
T0*+
_output_shapes
:????????? 2+
)sequential_5/time_distributed_2/Reshape_2?
%sequential_5/time_distributed_3/ShapeShape2sequential_5/time_distributed_2/Reshape_1:output:0*
T0*
_output_shapes
:2'
%sequential_5/time_distributed_3/Shape?
3sequential_5/time_distributed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential_5/time_distributed_3/strided_slice/stack?
5sequential_5/time_distributed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/time_distributed_3/strided_slice/stack_1?
5sequential_5/time_distributed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_5/time_distributed_3/strided_slice/stack_2?
-sequential_5/time_distributed_3/strided_sliceStridedSlice.sequential_5/time_distributed_3/Shape:output:0<sequential_5/time_distributed_3/strided_slice/stack:output:0>sequential_5/time_distributed_3/strided_slice/stack_1:output:0>sequential_5/time_distributed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_5/time_distributed_3/strided_slice?
-sequential_5/time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2/
-sequential_5/time_distributed_3/Reshape/shape?
'sequential_5/time_distributed_3/ReshapeReshape2sequential_5/time_distributed_2/Reshape_1:output:06sequential_5/time_distributed_3/Reshape/shape:output:0*
T0*+
_output_shapes
:????????? 2)
'sequential_5/time_distributed_3/Reshape?
/sequential_5/time_distributed_3/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   21
/sequential_5/time_distributed_3/flatten_1/Const?
1sequential_5/time_distributed_3/flatten_1/ReshapeReshape0sequential_5/time_distributed_3/Reshape:output:08sequential_5/time_distributed_3/flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????23
1sequential_5/time_distributed_3/flatten_1/Reshape?
1sequential_5/time_distributed_3/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????23
1sequential_5/time_distributed_3/Reshape_1/shape/0?
1sequential_5/time_distributed_3/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?23
1sequential_5/time_distributed_3/Reshape_1/shape/2?
/sequential_5/time_distributed_3/Reshape_1/shapePack:sequential_5/time_distributed_3/Reshape_1/shape/0:output:06sequential_5/time_distributed_3/strided_slice:output:0:sequential_5/time_distributed_3/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:21
/sequential_5/time_distributed_3/Reshape_1/shape?
)sequential_5/time_distributed_3/Reshape_1Reshape:sequential_5/time_distributed_3/flatten_1/Reshape:output:08sequential_5/time_distributed_3/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:???????????????????2+
)sequential_5/time_distributed_3/Reshape_1?
/sequential_5/time_distributed_3/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       21
/sequential_5/time_distributed_3/Reshape_2/shape?
)sequential_5/time_distributed_3/Reshape_2Reshape2sequential_5/time_distributed_2/Reshape_1:output:08sequential_5/time_distributed_3/Reshape_2/shape:output:0*
T0*+
_output_shapes
:????????? 2+
)sequential_5/time_distributed_3/Reshape_2?
sequential_5/lstm_5/ShapeShape2sequential_5/time_distributed_3/Reshape_1:output:0*
T0*
_output_shapes
:2
sequential_5/lstm_5/Shape?
'sequential_5/lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/lstm_5/strided_slice/stack?
)sequential_5/lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_5/lstm_5/strided_slice/stack_1?
)sequential_5/lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_5/lstm_5/strided_slice/stack_2?
!sequential_5/lstm_5/strided_sliceStridedSlice"sequential_5/lstm_5/Shape:output:00sequential_5/lstm_5/strided_slice/stack:output:02sequential_5/lstm_5/strided_slice/stack_1:output:02sequential_5/lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_5/lstm_5/strided_slice?
sequential_5/lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
sequential_5/lstm_5/zeros/mul/y?
sequential_5/lstm_5/zeros/mulMul*sequential_5/lstm_5/strided_slice:output:0(sequential_5/lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_5/lstm_5/zeros/mul?
 sequential_5/lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_5/lstm_5/zeros/Less/y?
sequential_5/lstm_5/zeros/LessLess!sequential_5/lstm_5/zeros/mul:z:0)sequential_5/lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_5/lstm_5/zeros/Less?
"sequential_5/lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"sequential_5/lstm_5/zeros/packed/1?
 sequential_5/lstm_5/zeros/packedPack*sequential_5/lstm_5/strided_slice:output:0+sequential_5/lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_5/lstm_5/zeros/packed?
sequential_5/lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_5/lstm_5/zeros/Const?
sequential_5/lstm_5/zerosFill)sequential_5/lstm_5/zeros/packed:output:0(sequential_5/lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
sequential_5/lstm_5/zeros?
!sequential_5/lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22#
!sequential_5/lstm_5/zeros_1/mul/y?
sequential_5/lstm_5/zeros_1/mulMul*sequential_5/lstm_5/strided_slice:output:0*sequential_5/lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_5/lstm_5/zeros_1/mul?
"sequential_5/lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_5/lstm_5/zeros_1/Less/y?
 sequential_5/lstm_5/zeros_1/LessLess#sequential_5/lstm_5/zeros_1/mul:z:0+sequential_5/lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_5/lstm_5/zeros_1/Less?
$sequential_5/lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22&
$sequential_5/lstm_5/zeros_1/packed/1?
"sequential_5/lstm_5/zeros_1/packedPack*sequential_5/lstm_5/strided_slice:output:0-sequential_5/lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_5/lstm_5/zeros_1/packed?
!sequential_5/lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_5/lstm_5/zeros_1/Const?
sequential_5/lstm_5/zeros_1Fill+sequential_5/lstm_5/zeros_1/packed:output:0*sequential_5/lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
sequential_5/lstm_5/zeros_1?
"sequential_5/lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_5/lstm_5/transpose/perm?
sequential_5/lstm_5/transpose	Transpose2sequential_5/time_distributed_3/Reshape_1:output:0+sequential_5/lstm_5/transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
sequential_5/lstm_5/transpose?
sequential_5/lstm_5/Shape_1Shape!sequential_5/lstm_5/transpose:y:0*
T0*
_output_shapes
:2
sequential_5/lstm_5/Shape_1?
)sequential_5/lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_5/lstm_5/strided_slice_1/stack?
+sequential_5/lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_5/strided_slice_1/stack_1?
+sequential_5/lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_5/strided_slice_1/stack_2?
#sequential_5/lstm_5/strided_slice_1StridedSlice$sequential_5/lstm_5/Shape_1:output:02sequential_5/lstm_5/strided_slice_1/stack:output:04sequential_5/lstm_5/strided_slice_1/stack_1:output:04sequential_5/lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_5/lstm_5/strided_slice_1?
/sequential_5/lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/sequential_5/lstm_5/TensorArrayV2/element_shape?
!sequential_5/lstm_5/TensorArrayV2TensorListReserve8sequential_5/lstm_5/TensorArrayV2/element_shape:output:0,sequential_5/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_5/lstm_5/TensorArrayV2?
Isequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2K
Isequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape?
;sequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_5/lstm_5/transpose:y:0Rsequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor?
)sequential_5/lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_5/lstm_5/strided_slice_2/stack?
+sequential_5/lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_5/strided_slice_2/stack_1?
+sequential_5/lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_5/strided_slice_2/stack_2?
#sequential_5/lstm_5/strided_slice_2StridedSlice!sequential_5/lstm_5/transpose:y:02sequential_5/lstm_5/strided_slice_2/stack:output:04sequential_5/lstm_5/strided_slice_2/stack_1:output:04sequential_5/lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2%
#sequential_5/lstm_5/strided_slice_2?
5sequential_5/lstm_5/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp>sequential_5_lstm_5_lstm_cell_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype027
5sequential_5/lstm_5/lstm_cell_7/MatMul/ReadVariableOp?
&sequential_5/lstm_5/lstm_cell_7/MatMulMatMul,sequential_5/lstm_5/strided_slice_2:output:0=sequential_5/lstm_5/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential_5/lstm_5/lstm_cell_7/MatMul?
7sequential_5/lstm_5/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp@sequential_5_lstm_5_lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype029
7sequential_5/lstm_5/lstm_cell_7/MatMul_1/ReadVariableOp?
(sequential_5/lstm_5/lstm_cell_7/MatMul_1MatMul"sequential_5/lstm_5/zeros:output:0?sequential_5/lstm_5/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(sequential_5/lstm_5/lstm_cell_7/MatMul_1?
#sequential_5/lstm_5/lstm_cell_7/addAddV20sequential_5/lstm_5/lstm_cell_7/MatMul:product:02sequential_5/lstm_5/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2%
#sequential_5/lstm_5/lstm_cell_7/add?
6sequential_5/lstm_5/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp?sequential_5_lstm_5_lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential_5/lstm_5/lstm_cell_7/BiasAdd/ReadVariableOp?
'sequential_5/lstm_5/lstm_cell_7/BiasAddBiasAdd'sequential_5/lstm_5/lstm_cell_7/add:z:0>sequential_5/lstm_5/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_5/lstm_5/lstm_cell_7/BiasAdd?
%sequential_5/lstm_5/lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_5/lstm_5/lstm_cell_7/Const?
/sequential_5/lstm_5/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_5/lstm_5/lstm_cell_7/split/split_dim?
%sequential_5/lstm_5/lstm_cell_7/splitSplit8sequential_5/lstm_5/lstm_cell_7/split/split_dim:output:00sequential_5/lstm_5/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2'
%sequential_5/lstm_5/lstm_cell_7/split?
'sequential_5/lstm_5/lstm_cell_7/SigmoidSigmoid.sequential_5/lstm_5/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22)
'sequential_5/lstm_5/lstm_cell_7/Sigmoid?
)sequential_5/lstm_5/lstm_cell_7/Sigmoid_1Sigmoid.sequential_5/lstm_5/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????22+
)sequential_5/lstm_5/lstm_cell_7/Sigmoid_1?
#sequential_5/lstm_5/lstm_cell_7/mulMul-sequential_5/lstm_5/lstm_cell_7/Sigmoid_1:y:0$sequential_5/lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:?????????22%
#sequential_5/lstm_5/lstm_cell_7/mul?
$sequential_5/lstm_5/lstm_cell_7/ReluRelu.sequential_5/lstm_5/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22&
$sequential_5/lstm_5/lstm_cell_7/Relu?
%sequential_5/lstm_5/lstm_cell_7/mul_1Mul+sequential_5/lstm_5/lstm_cell_7/Sigmoid:y:02sequential_5/lstm_5/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22'
%sequential_5/lstm_5/lstm_cell_7/mul_1?
%sequential_5/lstm_5/lstm_cell_7/add_1AddV2'sequential_5/lstm_5/lstm_cell_7/mul:z:0)sequential_5/lstm_5/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22'
%sequential_5/lstm_5/lstm_cell_7/add_1?
)sequential_5/lstm_5/lstm_cell_7/Sigmoid_2Sigmoid.sequential_5/lstm_5/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????22+
)sequential_5/lstm_5/lstm_cell_7/Sigmoid_2?
&sequential_5/lstm_5/lstm_cell_7/Relu_1Relu)sequential_5/lstm_5/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22(
&sequential_5/lstm_5/lstm_cell_7/Relu_1?
%sequential_5/lstm_5/lstm_cell_7/mul_2Mul-sequential_5/lstm_5/lstm_cell_7/Sigmoid_2:y:04sequential_5/lstm_5/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22'
%sequential_5/lstm_5/lstm_cell_7/mul_2?
1sequential_5/lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   23
1sequential_5/lstm_5/TensorArrayV2_1/element_shape?
#sequential_5/lstm_5/TensorArrayV2_1TensorListReserve:sequential_5/lstm_5/TensorArrayV2_1/element_shape:output:0,sequential_5/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_5/lstm_5/TensorArrayV2_1v
sequential_5/lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_5/lstm_5/time?
,sequential_5/lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,sequential_5/lstm_5/while/maximum_iterations?
&sequential_5/lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_5/lstm_5/while/loop_counter?
sequential_5/lstm_5/whileWhile/sequential_5/lstm_5/while/loop_counter:output:05sequential_5/lstm_5/while/maximum_iterations:output:0!sequential_5/lstm_5/time:output:0,sequential_5/lstm_5/TensorArrayV2_1:handle:0"sequential_5/lstm_5/zeros:output:0$sequential_5/lstm_5/zeros_1:output:0,sequential_5/lstm_5/strided_slice_1:output:0Ksequential_5/lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_5_lstm_5_lstm_cell_7_matmul_readvariableop_resource@sequential_5_lstm_5_lstm_cell_7_matmul_1_readvariableop_resource?sequential_5_lstm_5_lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*1
body)R'
%sequential_5_lstm_5_while_body_200329*1
cond)R'
%sequential_5_lstm_5_while_cond_200328*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
sequential_5/lstm_5/while?
Dsequential_5/lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2F
Dsequential_5/lstm_5/TensorArrayV2Stack/TensorListStack/element_shape?
6sequential_5/lstm_5/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_5/lstm_5/while:output:3Msequential_5/lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype028
6sequential_5/lstm_5/TensorArrayV2Stack/TensorListStack?
)sequential_5/lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)sequential_5/lstm_5/strided_slice_3/stack?
+sequential_5/lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_5/lstm_5/strided_slice_3/stack_1?
+sequential_5/lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_5/strided_slice_3/stack_2?
#sequential_5/lstm_5/strided_slice_3StridedSlice?sequential_5/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:02sequential_5/lstm_5/strided_slice_3/stack:output:04sequential_5/lstm_5/strided_slice_3/stack_1:output:04sequential_5/lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2%
#sequential_5/lstm_5/strided_slice_3?
$sequential_5/lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_5/lstm_5/transpose_1/perm?
sequential_5/lstm_5/transpose_1	Transpose?sequential_5/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_5/lstm_5/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22!
sequential_5/lstm_5/transpose_1?
sequential_5/lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_5/lstm_5/runtime?
sequential_5/lstm_6/ShapeShape#sequential_5/lstm_5/transpose_1:y:0*
T0*
_output_shapes
:2
sequential_5/lstm_6/Shape?
'sequential_5/lstm_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/lstm_6/strided_slice/stack?
)sequential_5/lstm_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_5/lstm_6/strided_slice/stack_1?
)sequential_5/lstm_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_5/lstm_6/strided_slice/stack_2?
!sequential_5/lstm_6/strided_sliceStridedSlice"sequential_5/lstm_6/Shape:output:00sequential_5/lstm_6/strided_slice/stack:output:02sequential_5/lstm_6/strided_slice/stack_1:output:02sequential_5/lstm_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_5/lstm_6/strided_slice?
sequential_5/lstm_6/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_5/lstm_6/zeros/mul/y?
sequential_5/lstm_6/zeros/mulMul*sequential_5/lstm_6/strided_slice:output:0(sequential_5/lstm_6/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_5/lstm_6/zeros/mul?
 sequential_5/lstm_6/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_5/lstm_6/zeros/Less/y?
sequential_5/lstm_6/zeros/LessLess!sequential_5/lstm_6/zeros/mul:z:0)sequential_5/lstm_6/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_5/lstm_6/zeros/Less?
"sequential_5/lstm_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_5/lstm_6/zeros/packed/1?
 sequential_5/lstm_6/zeros/packedPack*sequential_5/lstm_6/strided_slice:output:0+sequential_5/lstm_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_5/lstm_6/zeros/packed?
sequential_5/lstm_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_5/lstm_6/zeros/Const?
sequential_5/lstm_6/zerosFill)sequential_5/lstm_6/zeros/packed:output:0(sequential_5/lstm_6/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
sequential_5/lstm_6/zeros?
!sequential_5/lstm_6/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_5/lstm_6/zeros_1/mul/y?
sequential_5/lstm_6/zeros_1/mulMul*sequential_5/lstm_6/strided_slice:output:0*sequential_5/lstm_6/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_5/lstm_6/zeros_1/mul?
"sequential_5/lstm_6/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_5/lstm_6/zeros_1/Less/y?
 sequential_5/lstm_6/zeros_1/LessLess#sequential_5/lstm_6/zeros_1/mul:z:0+sequential_5/lstm_6/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_5/lstm_6/zeros_1/Less?
$sequential_5/lstm_6/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_5/lstm_6/zeros_1/packed/1?
"sequential_5/lstm_6/zeros_1/packedPack*sequential_5/lstm_6/strided_slice:output:0-sequential_5/lstm_6/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_5/lstm_6/zeros_1/packed?
!sequential_5/lstm_6/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_5/lstm_6/zeros_1/Const?
sequential_5/lstm_6/zeros_1Fill+sequential_5/lstm_6/zeros_1/packed:output:0*sequential_5/lstm_6/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
sequential_5/lstm_6/zeros_1?
"sequential_5/lstm_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_5/lstm_6/transpose/perm?
sequential_5/lstm_6/transpose	Transpose#sequential_5/lstm_5/transpose_1:y:0+sequential_5/lstm_6/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
sequential_5/lstm_6/transpose?
sequential_5/lstm_6/Shape_1Shape!sequential_5/lstm_6/transpose:y:0*
T0*
_output_shapes
:2
sequential_5/lstm_6/Shape_1?
)sequential_5/lstm_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_5/lstm_6/strided_slice_1/stack?
+sequential_5/lstm_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_6/strided_slice_1/stack_1?
+sequential_5/lstm_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_6/strided_slice_1/stack_2?
#sequential_5/lstm_6/strided_slice_1StridedSlice$sequential_5/lstm_6/Shape_1:output:02sequential_5/lstm_6/strided_slice_1/stack:output:04sequential_5/lstm_6/strided_slice_1/stack_1:output:04sequential_5/lstm_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_5/lstm_6/strided_slice_1?
/sequential_5/lstm_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/sequential_5/lstm_6/TensorArrayV2/element_shape?
!sequential_5/lstm_6/TensorArrayV2TensorListReserve8sequential_5/lstm_6/TensorArrayV2/element_shape:output:0,sequential_5/lstm_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_5/lstm_6/TensorArrayV2?
Isequential_5/lstm_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2K
Isequential_5/lstm_6/TensorArrayUnstack/TensorListFromTensor/element_shape?
;sequential_5/lstm_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_5/lstm_6/transpose:y:0Rsequential_5/lstm_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_5/lstm_6/TensorArrayUnstack/TensorListFromTensor?
)sequential_5/lstm_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_5/lstm_6/strided_slice_2/stack?
+sequential_5/lstm_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_6/strided_slice_2/stack_1?
+sequential_5/lstm_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_6/strided_slice_2/stack_2?
#sequential_5/lstm_6/strided_slice_2StridedSlice!sequential_5/lstm_6/transpose:y:02sequential_5/lstm_6/strided_slice_2/stack:output:04sequential_5/lstm_6/strided_slice_2/stack_1:output:04sequential_5/lstm_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2%
#sequential_5/lstm_6/strided_slice_2?
5sequential_5/lstm_6/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp>sequential_5_lstm_6_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype027
5sequential_5/lstm_6/lstm_cell_8/MatMul/ReadVariableOp?
&sequential_5/lstm_6/lstm_cell_8/MatMulMatMul,sequential_5/lstm_6/strided_slice_2:output:0=sequential_5/lstm_6/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2(
&sequential_5/lstm_6/lstm_cell_8/MatMul?
7sequential_5/lstm_6/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp@sequential_5_lstm_6_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype029
7sequential_5/lstm_6/lstm_cell_8/MatMul_1/ReadVariableOp?
(sequential_5/lstm_6/lstm_cell_8/MatMul_1MatMul"sequential_5/lstm_6/zeros:output:0?sequential_5/lstm_6/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2*
(sequential_5/lstm_6/lstm_cell_8/MatMul_1?
#sequential_5/lstm_6/lstm_cell_8/addAddV20sequential_5/lstm_6/lstm_cell_8/MatMul:product:02sequential_5/lstm_6/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2%
#sequential_5/lstm_6/lstm_cell_8/add?
6sequential_5/lstm_6/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp?sequential_5_lstm_6_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype028
6sequential_5/lstm_6/lstm_cell_8/BiasAdd/ReadVariableOp?
'sequential_5/lstm_6/lstm_cell_8/BiasAddBiasAdd'sequential_5/lstm_6/lstm_cell_8/add:z:0>sequential_5/lstm_6/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2)
'sequential_5/lstm_6/lstm_cell_8/BiasAdd?
%sequential_5/lstm_6/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_5/lstm_6/lstm_cell_8/Const?
/sequential_5/lstm_6/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_5/lstm_6/lstm_cell_8/split/split_dim?
%sequential_5/lstm_6/lstm_cell_8/splitSplit8sequential_5/lstm_6/lstm_cell_8/split/split_dim:output:00sequential_5/lstm_6/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2'
%sequential_5/lstm_6/lstm_cell_8/split?
'sequential_5/lstm_6/lstm_cell_8/SigmoidSigmoid.sequential_5/lstm_6/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2)
'sequential_5/lstm_6/lstm_cell_8/Sigmoid?
)sequential_5/lstm_6/lstm_cell_8/Sigmoid_1Sigmoid.sequential_5/lstm_6/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????2+
)sequential_5/lstm_6/lstm_cell_8/Sigmoid_1?
#sequential_5/lstm_6/lstm_cell_8/mulMul-sequential_5/lstm_6/lstm_cell_8/Sigmoid_1:y:0$sequential_5/lstm_6/zeros_1:output:0*
T0*'
_output_shapes
:?????????2%
#sequential_5/lstm_6/lstm_cell_8/mul?
$sequential_5/lstm_6/lstm_cell_8/ReluRelu.sequential_5/lstm_6/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2&
$sequential_5/lstm_6/lstm_cell_8/Relu?
%sequential_5/lstm_6/lstm_cell_8/mul_1Mul+sequential_5/lstm_6/lstm_cell_8/Sigmoid:y:02sequential_5/lstm_6/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2'
%sequential_5/lstm_6/lstm_cell_8/mul_1?
%sequential_5/lstm_6/lstm_cell_8/add_1AddV2'sequential_5/lstm_6/lstm_cell_8/mul:z:0)sequential_5/lstm_6/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2'
%sequential_5/lstm_6/lstm_cell_8/add_1?
)sequential_5/lstm_6/lstm_cell_8/Sigmoid_2Sigmoid.sequential_5/lstm_6/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????2+
)sequential_5/lstm_6/lstm_cell_8/Sigmoid_2?
&sequential_5/lstm_6/lstm_cell_8/Relu_1Relu)sequential_5/lstm_6/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2(
&sequential_5/lstm_6/lstm_cell_8/Relu_1?
%sequential_5/lstm_6/lstm_cell_8/mul_2Mul-sequential_5/lstm_6/lstm_cell_8/Sigmoid_2:y:04sequential_5/lstm_6/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2'
%sequential_5/lstm_6/lstm_cell_8/mul_2?
1sequential_5/lstm_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1sequential_5/lstm_6/TensorArrayV2_1/element_shape?
#sequential_5/lstm_6/TensorArrayV2_1TensorListReserve:sequential_5/lstm_6/TensorArrayV2_1/element_shape:output:0,sequential_5/lstm_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_5/lstm_6/TensorArrayV2_1v
sequential_5/lstm_6/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_5/lstm_6/time?
,sequential_5/lstm_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,sequential_5/lstm_6/while/maximum_iterations?
&sequential_5/lstm_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_5/lstm_6/while/loop_counter?
sequential_5/lstm_6/whileWhile/sequential_5/lstm_6/while/loop_counter:output:05sequential_5/lstm_6/while/maximum_iterations:output:0!sequential_5/lstm_6/time:output:0,sequential_5/lstm_6/TensorArrayV2_1:handle:0"sequential_5/lstm_6/zeros:output:0$sequential_5/lstm_6/zeros_1:output:0,sequential_5/lstm_6/strided_slice_1:output:0Ksequential_5/lstm_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_5_lstm_6_lstm_cell_8_matmul_readvariableop_resource@sequential_5_lstm_6_lstm_cell_8_matmul_1_readvariableop_resource?sequential_5_lstm_6_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*1
body)R'
%sequential_5_lstm_6_while_body_200478*1
cond)R'
%sequential_5_lstm_6_while_cond_200477*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
sequential_5/lstm_6/while?
Dsequential_5/lstm_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dsequential_5/lstm_6/TensorArrayV2Stack/TensorListStack/element_shape?
6sequential_5/lstm_6/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_5/lstm_6/while:output:3Msequential_5/lstm_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype028
6sequential_5/lstm_6/TensorArrayV2Stack/TensorListStack?
)sequential_5/lstm_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)sequential_5/lstm_6/strided_slice_3/stack?
+sequential_5/lstm_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_5/lstm_6/strided_slice_3/stack_1?
+sequential_5/lstm_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_6/strided_slice_3/stack_2?
#sequential_5/lstm_6/strided_slice_3StridedSlice?sequential_5/lstm_6/TensorArrayV2Stack/TensorListStack:tensor:02sequential_5/lstm_6/strided_slice_3/stack:output:04sequential_5/lstm_6/strided_slice_3/stack_1:output:04sequential_5/lstm_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2%
#sequential_5/lstm_6/strided_slice_3?
$sequential_5/lstm_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_5/lstm_6/transpose_1/perm?
sequential_5/lstm_6/transpose_1	Transpose?sequential_5/lstm_6/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_5/lstm_6/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2!
sequential_5/lstm_6/transpose_1?
sequential_5/lstm_6/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_5/lstm_6/runtime?
+sequential_5/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+sequential_5/dense_12/MatMul/ReadVariableOp?
sequential_5/dense_12/MatMulMatMul,sequential_5/lstm_6/strided_slice_3:output:03sequential_5/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_5/dense_12/MatMul?
,sequential_5/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_5/dense_12/BiasAdd/ReadVariableOp?
sequential_5/dense_12/BiasAddBiasAdd&sequential_5/dense_12/MatMul:product:04sequential_5/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_5/dense_12/BiasAdd?
IdentityIdentity&sequential_5/dense_12/BiasAdd:output:0^sequential_5/lstm_5/while^sequential_5/lstm_6/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::26
sequential_5/lstm_5/whilesequential_5/lstm_5/while26
sequential_5/lstm_6/whilesequential_5/lstm_6/while:p l
8
_output_shapes&
$:"??????????????????

0
_user_specified_nametime_distributed_input
?/
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_203173

inputs
time_distributed_203133
time_distributed_203135
time_distributed_1_203140
time_distributed_1_203142
lstm_5_203153
lstm_5_203155
lstm_5_203157
lstm_6_203160
lstm_6_203162
lstm_6_203164
dense_12_203167
dense_12_203169
identity?? dense_12/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?lstm_6/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_203133time_distributed_203135*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"??????????????????	@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_2006932*
(time_distributed/StatefulPartitionedCall?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????
      2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????
2
time_distributed/Reshape?
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_203140time_distributed_1_203142*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"?????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_2008242,
*time_distributed_1/StatefulPartitionedCall?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????	   @   2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????	@2
time_distributed_1/Reshape?
"time_distributed_2/PartitionedCallPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_2009232$
"time_distributed_2/PartitionedCall?
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2"
 time_distributed_2/Reshape/shape?
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_2/Reshape?
"time_distributed_3/PartitionedCallPartitionedCall+time_distributed_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_2010122$
"time_distributed_3/PartitionedCall?
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2"
 time_distributed_3/Reshape/shape?
time_distributed_3/ReshapeReshape+time_distributed_2/PartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_3/Reshape?
lstm_5/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0lstm_5_203153lstm_5_203155lstm_5_203157*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_2026202 
lstm_5/StatefulPartitionedCall?
lstm_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0lstm_6_203160lstm_6_203162lstm_6_203164*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_6_layer_call_and_return_conditional_losses_2029552 
lstm_6/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall'lstm_6/StatefulPartitionedCall:output:0dense_12_203167dense_12_203169*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_2029952"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall^lstm_6/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall2@
lstm_6/StatefulPartitionedCalllstm_6/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:` \
8
_output_shapes&
$:"??????????????????

 
_user_specified_nameinputs
?0
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_203012
time_distributed_input
time_distributed_202257
time_distributed_202259
time_distributed_1_202282
time_distributed_1_202284
lstm_5_202643
lstm_5_202645
lstm_5_202647
lstm_6_202978
lstm_6_202980
lstm_6_202982
dense_12_203006
dense_12_203008
identity?? dense_12/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?lstm_6/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCalltime_distributed_inputtime_distributed_202257time_distributed_202259*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"??????????????????	@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_2006632*
(time_distributed/StatefulPartitionedCall?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????
      2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshapetime_distributed_input'time_distributed/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????
2
time_distributed/Reshape?
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_202282time_distributed_1_202284*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"?????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_2007942,
*time_distributed_1/StatefulPartitionedCall?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????	   @   2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????	@2
time_distributed_1/Reshape?
"time_distributed_2/PartitionedCallPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_2009012$
"time_distributed_2/PartitionedCall?
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2"
 time_distributed_2/Reshape/shape?
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_2/Reshape?
"time_distributed_3/PartitionedCallPartitionedCall+time_distributed_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_2009912$
"time_distributed_3/PartitionedCall?
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2"
 time_distributed_3/Reshape/shape?
time_distributed_3/ReshapeReshape+time_distributed_2/PartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_3/Reshape?
lstm_5/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0lstm_5_202643lstm_5_202645lstm_5_202647*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_2024672 
lstm_5/StatefulPartitionedCall?
lstm_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0lstm_6_202978lstm_6_202980lstm_6_202982*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_6_layer_call_and_return_conditional_losses_2028022 
lstm_6/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall'lstm_6/StatefulPartitionedCall:output:0dense_12_203006dense_12_203008*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_2029952"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall^lstm_6/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall2@
lstm_6/StatefulPartitionedCalllstm_6/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:p l
8
_output_shapes&
$:"??????????????????

0
_user_specified_nametime_distributed_input
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_204139

inputs8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slices
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????
      2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????
2	
Reshape?
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDimsReshape:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????	@*
paddingVALID*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????	@*
squeeze_dims

?????????2
conv1d_2/conv1d/Squeeze?
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp?
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????	@2
conv1d_2/BiasAddw
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????	@2
conv1d_2/Reluq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape_1/shape/3?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshapeconv1d_2/Relu:activations:0Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????	@2
	Reshape_1w
IdentityIdentityReshape_1:output:0*
T0*8
_output_shapes&
$:"??????????????????	@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:"??????????????????
:::` \
8
_output_shapes&
$:"??????????????????

 
_user_specified_nameinputs
?9
?
while_body_204721
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_7_matmul_readvariableop_resource_08
4while_lstm_cell_7_matmul_1_readvariableop_resource_07
3while_lstm_cell_7_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_7_matmul_readvariableop_resource6
2while_lstm_cell_7_matmul_1_readvariableop_resource5
1while_lstm_cell_7_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/lstm_cell_7/MatMul/ReadVariableOp?
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/MatMul?
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_7/MatMul_1/ReadVariableOp?
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/MatMul_1?
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/add?
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_7/BiasAdd/ReadVariableOp?
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/BiasAddt
while/lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_7/Const?
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_7/split/split_dim?
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_7/split?
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid?
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid_1?
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul?
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Relu?
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul_1?
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/add_1?
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid_2?
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Relu_1?
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?B
?
lstm_6_while_body_203541*
&lstm_6_while_lstm_6_while_loop_counter0
,lstm_6_while_lstm_6_while_maximum_iterations
lstm_6_while_placeholder
lstm_6_while_placeholder_1
lstm_6_while_placeholder_2
lstm_6_while_placeholder_3)
%lstm_6_while_lstm_6_strided_slice_1_0e
alstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_6_while_lstm_cell_8_matmul_readvariableop_resource_0?
;lstm_6_while_lstm_cell_8_matmul_1_readvariableop_resource_0>
:lstm_6_while_lstm_cell_8_biasadd_readvariableop_resource_0
lstm_6_while_identity
lstm_6_while_identity_1
lstm_6_while_identity_2
lstm_6_while_identity_3
lstm_6_while_identity_4
lstm_6_while_identity_5'
#lstm_6_while_lstm_6_strided_slice_1c
_lstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensor;
7lstm_6_while_lstm_cell_8_matmul_readvariableop_resource=
9lstm_6_while_lstm_cell_8_matmul_1_readvariableop_resource<
8lstm_6_while_lstm_cell_8_biasadd_readvariableop_resource??
>lstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2@
>lstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensor_0lstm_6_while_placeholderGlstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype022
0lstm_6/while/TensorArrayV2Read/TensorListGetItem?
.lstm_6/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp9lstm_6_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype020
.lstm_6/while/lstm_cell_8/MatMul/ReadVariableOp?
lstm_6/while/lstm_cell_8/MatMulMatMul7lstm_6/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_6/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
lstm_6/while/lstm_cell_8/MatMul?
0lstm_6/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp;lstm_6_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype022
0lstm_6/while/lstm_cell_8/MatMul_1/ReadVariableOp?
!lstm_6/while/lstm_cell_8/MatMul_1MatMullstm_6_while_placeholder_28lstm_6/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2#
!lstm_6/while/lstm_cell_8/MatMul_1?
lstm_6/while/lstm_cell_8/addAddV2)lstm_6/while/lstm_cell_8/MatMul:product:0+lstm_6/while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_6/while/lstm_cell_8/add?
/lstm_6/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp:lstm_6_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype021
/lstm_6/while/lstm_cell_8/BiasAdd/ReadVariableOp?
 lstm_6/while/lstm_cell_8/BiasAddBiasAdd lstm_6/while/lstm_cell_8/add:z:07lstm_6/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2"
 lstm_6/while/lstm_cell_8/BiasAdd?
lstm_6/while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_6/while/lstm_cell_8/Const?
(lstm_6/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_6/while/lstm_cell_8/split/split_dim?
lstm_6/while/lstm_cell_8/splitSplit1lstm_6/while/lstm_cell_8/split/split_dim:output:0)lstm_6/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2 
lstm_6/while/lstm_cell_8/split?
 lstm_6/while/lstm_cell_8/SigmoidSigmoid'lstm_6/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2"
 lstm_6/while/lstm_cell_8/Sigmoid?
"lstm_6/while/lstm_cell_8/Sigmoid_1Sigmoid'lstm_6/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????2$
"lstm_6/while/lstm_cell_8/Sigmoid_1?
lstm_6/while/lstm_cell_8/mulMul&lstm_6/while/lstm_cell_8/Sigmoid_1:y:0lstm_6_while_placeholder_3*
T0*'
_output_shapes
:?????????2
lstm_6/while/lstm_cell_8/mul?
lstm_6/while/lstm_cell_8/ReluRelu'lstm_6/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_6/while/lstm_cell_8/Relu?
lstm_6/while/lstm_cell_8/mul_1Mul$lstm_6/while/lstm_cell_8/Sigmoid:y:0+lstm_6/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2 
lstm_6/while/lstm_cell_8/mul_1?
lstm_6/while/lstm_cell_8/add_1AddV2 lstm_6/while/lstm_cell_8/mul:z:0"lstm_6/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2 
lstm_6/while/lstm_cell_8/add_1?
"lstm_6/while/lstm_cell_8/Sigmoid_2Sigmoid'lstm_6/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????2$
"lstm_6/while/lstm_cell_8/Sigmoid_2?
lstm_6/while/lstm_cell_8/Relu_1Relu"lstm_6/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2!
lstm_6/while/lstm_cell_8/Relu_1?
lstm_6/while/lstm_cell_8/mul_2Mul&lstm_6/while/lstm_cell_8/Sigmoid_2:y:0-lstm_6/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2 
lstm_6/while/lstm_cell_8/mul_2?
1lstm_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_6_while_placeholder_1lstm_6_while_placeholder"lstm_6/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_6/while/TensorArrayV2Write/TensorListSetItemj
lstm_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/while/add/y?
lstm_6/while/addAddV2lstm_6_while_placeholderlstm_6/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_6/while/addn
lstm_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/while/add_1/y?
lstm_6/while/add_1AddV2&lstm_6_while_lstm_6_while_loop_counterlstm_6/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_6/while/add_1s
lstm_6/while/IdentityIdentitylstm_6/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_6/while/Identity?
lstm_6/while/Identity_1Identity,lstm_6_while_lstm_6_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_6/while/Identity_1u
lstm_6/while/Identity_2Identitylstm_6/while/add:z:0*
T0*
_output_shapes
: 2
lstm_6/while/Identity_2?
lstm_6/while/Identity_3IdentityAlstm_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_6/while/Identity_3?
lstm_6/while/Identity_4Identity"lstm_6/while/lstm_cell_8/mul_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_6/while/Identity_4?
lstm_6/while/Identity_5Identity"lstm_6/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_6/while/Identity_5"7
lstm_6_while_identitylstm_6/while/Identity:output:0";
lstm_6_while_identity_1 lstm_6/while/Identity_1:output:0";
lstm_6_while_identity_2 lstm_6/while/Identity_2:output:0";
lstm_6_while_identity_3 lstm_6/while/Identity_3:output:0";
lstm_6_while_identity_4 lstm_6/while/Identity_4:output:0";
lstm_6_while_identity_5 lstm_6/while/Identity_5:output:0"L
#lstm_6_while_lstm_6_strided_slice_1%lstm_6_while_lstm_6_strided_slice_1_0"v
8lstm_6_while_lstm_cell_8_biasadd_readvariableop_resource:lstm_6_while_lstm_cell_8_biasadd_readvariableop_resource_0"x
9lstm_6_while_lstm_cell_8_matmul_1_readvariableop_resource;lstm_6_while_lstm_cell_8_matmul_1_readvariableop_resource_0"t
7lstm_6_while_lstm_cell_8_matmul_readvariableop_resource9lstm_6_while_lstm_cell_8_matmul_readvariableop_resource_0"?
_lstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensoralstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_204315

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slices
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:????????? 2	
Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten_1/Const?
flatten_1/ReshapeReshapeReshape:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshapeq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape_1/shape/2?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshapeflatten_1/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:???????????????????2
	Reshape_1t
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:"?????????????????? :` \
8
_output_shapes&
$:"?????????????????? 
 
_user_specified_nameinputs
?
?
while_cond_202716
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_202716___redundant_placeholder04
0while_while_cond_202716___redundant_placeholder14
0while_while_cond_202716___redundant_placeholder24
0while_while_cond_202716___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_202534
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_202534___redundant_placeholder04
0while_while_cond_202534___redundant_placeholder14
0while_while_cond_202534___redundant_placeholder24
0while_while_cond_202534___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
'__inference_lstm_5_layer_call_fn_204970

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_2024672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling1d_1_layer_call_fn_200846

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_2008402
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_204185

inputs8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slices
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????	   @   2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????	@2	
Reshape?
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_3/conv1d/ExpandDims/dim?
conv1d_3/conv1d/ExpandDims
ExpandDimsReshape:output:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????	@2
conv1d_3/conv1d/ExpandDims?
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dim?
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d_3/conv1d/ExpandDims_1?
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d_3/conv1d?
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d_3/conv1d/Squeeze?
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_3/BiasAdd/ReadVariableOp?
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
conv1d_3/BiasAddw
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
conv1d_3/Reluq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/3?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshapeconv1d_3/Relu:activations:0Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2
	Reshape_1w
IdentityIdentityReshape_1:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:"??????????????????	@:::` \
8
_output_shapes&
$:"??????????????????	@
 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_201088

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:??????????:?????????2:?????????2::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates
?	
?
lstm_5_while_cond_203784*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3,
(lstm_5_while_less_lstm_5_strided_slice_1B
>lstm_5_while_lstm_5_while_cond_203784___redundant_placeholder0B
>lstm_5_while_lstm_5_while_cond_203784___redundant_placeholder1B
>lstm_5_while_lstm_5_while_cond_203784___redundant_placeholder2B
>lstm_5_while_lstm_5_while_cond_203784___redundant_placeholder3
lstm_5_while_identity
?
lstm_5/while/LessLesslstm_5_while_placeholder(lstm_5_while_less_lstm_5_strided_slice_1*
T0*
_output_shapes
: 2
lstm_5/while/Lessr
lstm_5/while/IdentityIdentitylstm_5/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_5/while/Identity"7
lstm_5_while_identitylstm_5/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
'__inference_lstm_6_layer_call_fn_205637
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_6_layer_call_and_return_conditional_losses_2022262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????2
"
_user_specified_name
inputs/0
?9
?
while_body_204546
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_7_matmul_readvariableop_resource_08
4while_lstm_cell_7_matmul_1_readvariableop_resource_07
3while_lstm_cell_7_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_7_matmul_readvariableop_resource6
2while_lstm_cell_7_matmul_1_readvariableop_resource5
1while_lstm_cell_7_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/lstm_cell_7/MatMul/ReadVariableOp?
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/MatMul?
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_7/MatMul_1/ReadVariableOp?
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/MatMul_1?
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/add?
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_7/BiasAdd/ReadVariableOp?
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/BiasAddt
while/lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_7/Const?
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_7/split/split_dim?
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_7/split?
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid?
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid_1?
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul?
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Relu?
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul_1?
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/add_1?
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid_2?
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Relu_1?
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_200794

inputs
conv1d_3_200783
conv1d_3_200785
identity?? conv1d_3/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slices
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????	   @   2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????	@2	
Reshape?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv1d_3_200783conv1d_3_200785*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_2007272"
 conv1d_3/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/3?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape)conv1d_3/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2
	Reshape_1?
IdentityIdentityReshape_1:output:0!^conv1d_3/StatefulPartitionedCall*
T0*8
_output_shapes&
$:"?????????????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:"??????????????????	@::2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall:` \
8
_output_shapes&
$:"??????????????????	@
 
_user_specified_nameinputs
?
?
%sequential_5_lstm_5_while_cond_200328D
@sequential_5_lstm_5_while_sequential_5_lstm_5_while_loop_counterJ
Fsequential_5_lstm_5_while_sequential_5_lstm_5_while_maximum_iterations)
%sequential_5_lstm_5_while_placeholder+
'sequential_5_lstm_5_while_placeholder_1+
'sequential_5_lstm_5_while_placeholder_2+
'sequential_5_lstm_5_while_placeholder_3F
Bsequential_5_lstm_5_while_less_sequential_5_lstm_5_strided_slice_1\
Xsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_200328___redundant_placeholder0\
Xsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_200328___redundant_placeholder1\
Xsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_200328___redundant_placeholder2\
Xsequential_5_lstm_5_while_sequential_5_lstm_5_while_cond_200328___redundant_placeholder3&
"sequential_5_lstm_5_while_identity
?
sequential_5/lstm_5/while/LessLess%sequential_5_lstm_5_while_placeholderBsequential_5_lstm_5_while_less_sequential_5_lstm_5_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_5/lstm_5/while/Less?
"sequential_5/lstm_5/while/IdentityIdentity"sequential_5/lstm_5/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_5/lstm_5/while/Identity"Q
"sequential_5_lstm_5_while_identity+sequential_5/lstm_5/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?9
?
while_body_205049
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_8_matmul_readvariableop_resource_08
4while_lstm_cell_8_matmul_1_readvariableop_resource_07
3while_lstm_cell_8_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_8_matmul_readvariableop_resource6
2while_lstm_cell_8_matmul_1_readvariableop_resource5
1while_lstm_cell_8_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp?
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/MatMul?
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp?
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/MatMul_1?
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/add?
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp?
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/BiasAddt
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_8/Const?
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_8/split?
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid?
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid_1?
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul?
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Relu?
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul_1?
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/add_1?
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid_2?
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Relu_1?
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?W
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_202467

inputs.
*lstm_cell_7_matmul_readvariableop_resource0
,lstm_cell_7_matmul_1_readvariableop_resource/
+lstm_cell_7_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!lstm_cell_7/MatMul/ReadVariableOp?
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/MatMul?
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_7/MatMul_1/ReadVariableOp?
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/MatMul_1?
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/add?
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_7/BiasAdd/ReadVariableOp?
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/BiasAddh
lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/Const|
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/split/split_dim?
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_7/split?
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid?
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid_1?
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mulz
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Relu?
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mul_1?
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/add_1?
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid_2y
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Relu_1?
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_202382*
condR
while_cond_202381*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?0
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_203055
time_distributed_input
time_distributed_203015
time_distributed_203017
time_distributed_1_203022
time_distributed_1_203024
lstm_5_203035
lstm_5_203037
lstm_5_203039
lstm_6_203042
lstm_6_203044
lstm_6_203046
dense_12_203049
dense_12_203051
identity?? dense_12/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?lstm_6/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCalltime_distributed_inputtime_distributed_203015time_distributed_203017*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"??????????????????	@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_2006932*
(time_distributed/StatefulPartitionedCall?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????
      2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshapetime_distributed_input'time_distributed/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????
2
time_distributed/Reshape?
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_203022time_distributed_1_203024*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"?????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_2008242,
*time_distributed_1/StatefulPartitionedCall?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????	   @   2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????	@2
time_distributed_1/Reshape?
"time_distributed_2/PartitionedCallPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_2009232$
"time_distributed_2/PartitionedCall?
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2"
 time_distributed_2/Reshape/shape?
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_2/Reshape?
"time_distributed_3/PartitionedCallPartitionedCall+time_distributed_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_2010122$
"time_distributed_3/PartitionedCall?
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2"
 time_distributed_3/Reshape/shape?
time_distributed_3/ReshapeReshape+time_distributed_2/PartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_3/Reshape?
lstm_5/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0lstm_5_203035lstm_5_203037lstm_5_203039*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_2026202 
lstm_5/StatefulPartitionedCall?
lstm_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0lstm_6_203042lstm_6_203044lstm_6_203046*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_6_layer_call_and_return_conditional_losses_2029552 
lstm_6/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall'lstm_6/StatefulPartitionedCall:output:0dense_12_203049dense_12_203051*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_2029952"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall^lstm_6/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall2@
lstm_6/StatefulPartitionedCalllstm_6/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:p l
8
_output_shapes&
$:"??????????????????

0
_user_specified_nametime_distributed_input
?
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_204298

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slices
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:????????? 2	
Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten_1/Const?
flatten_1/ReshapeReshapeReshape:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshapeq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape_1/shape/2?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshapeflatten_1/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:???????????????????2
	Reshape_1t
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:"?????????????????? :` \
8
_output_shapes&
$:"?????????????????? 
 
_user_specified_nameinputs
?
F
*__inference_flatten_1_layer_call_fn_205717

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2009432
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
while_cond_205201
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_205201___redundant_placeholder04
0while_while_cond_205201___redundant_placeholder14
0while_while_cond_205201___redundant_placeholder24
0while_while_cond_205201___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?W
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_204806

inputs.
*lstm_cell_7_matmul_readvariableop_resource0
,lstm_cell_7_matmul_1_readvariableop_resource/
+lstm_cell_7_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!lstm_cell_7/MatMul/ReadVariableOp?
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/MatMul?
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_7/MatMul_1/ReadVariableOp?
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/MatMul_1?
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/add?
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_7/BiasAdd/ReadVariableOp?
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/BiasAddh
lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/Const|
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/split/split_dim?
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_7/split?
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid?
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid_1?
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mulz
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Relu?
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mul_1?
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/add_1?
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid_2y
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Relu_1?
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_204721*
condR
while_cond_204720*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?	
?
lstm_5_while_cond_203391*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3,
(lstm_5_while_less_lstm_5_strided_slice_1B
>lstm_5_while_lstm_5_while_cond_203391___redundant_placeholder0B
>lstm_5_while_lstm_5_while_cond_203391___redundant_placeholder1B
>lstm_5_while_lstm_5_while_cond_203391___redundant_placeholder2B
>lstm_5_while_lstm_5_while_cond_203391___redundant_placeholder3
lstm_5_while_identity
?
lstm_5/while/LessLesslstm_5_while_placeholder(lstm_5_while_less_lstm_5_strided_slice_1*
T0*
_output_shapes
: 2
lstm_5/while/Lessr
lstm_5/while/IdentityIdentitylstm_5/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_5/while/Identity"7
lstm_5_while_identitylstm_5/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?D
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_201616

inputs
lstm_cell_7_201534
lstm_cell_7_201536
lstm_cell_7_201538
identity??#lstm_cell_7/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_7/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_7_201534lstm_cell_7_201536lstm_cell_7_201538*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_2011212%
#lstm_cell_7/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_7_201534lstm_cell_7_201536lstm_cell_7_201538*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_201547*
condR
while_cond_201546*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_7/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2J
#lstm_cell_7/StatefulPartitionedCall#lstm_cell_7/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?	
?
-__inference_sequential_5_layer_call_fn_204054

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_2031012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
8
_output_shapes&
$:"??????????????????

 
_user_specified_nameinputs
?9
?
while_body_205377
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_8_matmul_readvariableop_resource_08
4while_lstm_cell_8_matmul_1_readvariableop_resource_07
3while_lstm_cell_8_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_8_matmul_readvariableop_resource6
2while_lstm_cell_8_matmul_1_readvariableop_resource5
1while_lstm_cell_8_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp?
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/MatMul?
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp?
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/MatMul_1?
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/add?
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp?
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/BiasAddt
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_8/Const?
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_8/split?
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid?
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid_1?
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul?
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Relu?
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul_1?
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/add_1?
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid_2?
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Relu_1?
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_lstm_cell_8_layer_call_fn_205900

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_2016982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????2:?????????:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?
?
'__inference_lstm_6_layer_call_fn_205626
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_6_layer_call_and_return_conditional_losses_2020942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????2
"
_user_specified_name
inputs/0
?
?
3__inference_time_distributed_1_layer_call_fn_204231

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"?????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_2008242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*8
_output_shapes&
$:"?????????????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:"??????????????????	@::22
StatefulPartitionedCallStatefulPartitionedCall:` \
8
_output_shapes&
$:"??????????????????	@
 
_user_specified_nameinputs
?
?
while_cond_205048
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_205048___redundant_placeholder04
0while_while_cond_205048___redundant_placeholder14
0while_while_cond_205048___redundant_placeholder24
0while_while_cond_205048___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
,__inference_lstm_cell_7_layer_call_fn_205817

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_2011212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:??????????:?????????2:?????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
?
?
'__inference_lstm_5_layer_call_fn_204981

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_2026202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?R
?
%sequential_5_lstm_6_while_body_200478D
@sequential_5_lstm_6_while_sequential_5_lstm_6_while_loop_counterJ
Fsequential_5_lstm_6_while_sequential_5_lstm_6_while_maximum_iterations)
%sequential_5_lstm_6_while_placeholder+
'sequential_5_lstm_6_while_placeholder_1+
'sequential_5_lstm_6_while_placeholder_2+
'sequential_5_lstm_6_while_placeholder_3C
?sequential_5_lstm_6_while_sequential_5_lstm_6_strided_slice_1_0
{sequential_5_lstm_6_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_6_tensorarrayunstack_tensorlistfromtensor_0J
Fsequential_5_lstm_6_while_lstm_cell_8_matmul_readvariableop_resource_0L
Hsequential_5_lstm_6_while_lstm_cell_8_matmul_1_readvariableop_resource_0K
Gsequential_5_lstm_6_while_lstm_cell_8_biasadd_readvariableop_resource_0&
"sequential_5_lstm_6_while_identity(
$sequential_5_lstm_6_while_identity_1(
$sequential_5_lstm_6_while_identity_2(
$sequential_5_lstm_6_while_identity_3(
$sequential_5_lstm_6_while_identity_4(
$sequential_5_lstm_6_while_identity_5A
=sequential_5_lstm_6_while_sequential_5_lstm_6_strided_slice_1}
ysequential_5_lstm_6_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_6_tensorarrayunstack_tensorlistfromtensorH
Dsequential_5_lstm_6_while_lstm_cell_8_matmul_readvariableop_resourceJ
Fsequential_5_lstm_6_while_lstm_cell_8_matmul_1_readvariableop_resourceI
Esequential_5_lstm_6_while_lstm_cell_8_biasadd_readvariableop_resource??
Ksequential_5/lstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2M
Ksequential_5/lstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=sequential_5/lstm_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_5_lstm_6_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_6_tensorarrayunstack_tensorlistfromtensor_0%sequential_5_lstm_6_while_placeholderTsequential_5/lstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02?
=sequential_5/lstm_6/while/TensorArrayV2Read/TensorListGetItem?
;sequential_5/lstm_6/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOpFsequential_5_lstm_6_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype02=
;sequential_5/lstm_6/while/lstm_cell_8/MatMul/ReadVariableOp?
,sequential_5/lstm_6/while/lstm_cell_8/MatMulMatMulDsequential_5/lstm_6/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_5/lstm_6/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2.
,sequential_5/lstm_6/while/lstm_cell_8/MatMul?
=sequential_5/lstm_6/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOpHsequential_5_lstm_6_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype02?
=sequential_5/lstm_6/while/lstm_cell_8/MatMul_1/ReadVariableOp?
.sequential_5/lstm_6/while/lstm_cell_8/MatMul_1MatMul'sequential_5_lstm_6_while_placeholder_2Esequential_5/lstm_6/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d20
.sequential_5/lstm_6/while/lstm_cell_8/MatMul_1?
)sequential_5/lstm_6/while/lstm_cell_8/addAddV26sequential_5/lstm_6/while/lstm_cell_8/MatMul:product:08sequential_5/lstm_6/while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2+
)sequential_5/lstm_6/while/lstm_cell_8/add?
<sequential_5/lstm_6/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOpGsequential_5_lstm_6_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype02>
<sequential_5/lstm_6/while/lstm_cell_8/BiasAdd/ReadVariableOp?
-sequential_5/lstm_6/while/lstm_cell_8/BiasAddBiasAdd-sequential_5/lstm_6/while/lstm_cell_8/add:z:0Dsequential_5/lstm_6/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2/
-sequential_5/lstm_6/while/lstm_cell_8/BiasAdd?
+sequential_5/lstm_6/while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2-
+sequential_5/lstm_6/while/lstm_cell_8/Const?
5sequential_5/lstm_6/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_5/lstm_6/while/lstm_cell_8/split/split_dim?
+sequential_5/lstm_6/while/lstm_cell_8/splitSplit>sequential_5/lstm_6/while/lstm_cell_8/split/split_dim:output:06sequential_5/lstm_6/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2-
+sequential_5/lstm_6/while/lstm_cell_8/split?
-sequential_5/lstm_6/while/lstm_cell_8/SigmoidSigmoid4sequential_5/lstm_6/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2/
-sequential_5/lstm_6/while/lstm_cell_8/Sigmoid?
/sequential_5/lstm_6/while/lstm_cell_8/Sigmoid_1Sigmoid4sequential_5/lstm_6/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????21
/sequential_5/lstm_6/while/lstm_cell_8/Sigmoid_1?
)sequential_5/lstm_6/while/lstm_cell_8/mulMul3sequential_5/lstm_6/while/lstm_cell_8/Sigmoid_1:y:0'sequential_5_lstm_6_while_placeholder_3*
T0*'
_output_shapes
:?????????2+
)sequential_5/lstm_6/while/lstm_cell_8/mul?
*sequential_5/lstm_6/while/lstm_cell_8/ReluRelu4sequential_5/lstm_6/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2,
*sequential_5/lstm_6/while/lstm_cell_8/Relu?
+sequential_5/lstm_6/while/lstm_cell_8/mul_1Mul1sequential_5/lstm_6/while/lstm_cell_8/Sigmoid:y:08sequential_5/lstm_6/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2-
+sequential_5/lstm_6/while/lstm_cell_8/mul_1?
+sequential_5/lstm_6/while/lstm_cell_8/add_1AddV2-sequential_5/lstm_6/while/lstm_cell_8/mul:z:0/sequential_5/lstm_6/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2-
+sequential_5/lstm_6/while/lstm_cell_8/add_1?
/sequential_5/lstm_6/while/lstm_cell_8/Sigmoid_2Sigmoid4sequential_5/lstm_6/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????21
/sequential_5/lstm_6/while/lstm_cell_8/Sigmoid_2?
,sequential_5/lstm_6/while/lstm_cell_8/Relu_1Relu/sequential_5/lstm_6/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2.
,sequential_5/lstm_6/while/lstm_cell_8/Relu_1?
+sequential_5/lstm_6/while/lstm_cell_8/mul_2Mul3sequential_5/lstm_6/while/lstm_cell_8/Sigmoid_2:y:0:sequential_5/lstm_6/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2-
+sequential_5/lstm_6/while/lstm_cell_8/mul_2?
>sequential_5/lstm_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_5_lstm_6_while_placeholder_1%sequential_5_lstm_6_while_placeholder/sequential_5/lstm_6/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype02@
>sequential_5/lstm_6/while/TensorArrayV2Write/TensorListSetItem?
sequential_5/lstm_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_5/lstm_6/while/add/y?
sequential_5/lstm_6/while/addAddV2%sequential_5_lstm_6_while_placeholder(sequential_5/lstm_6/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_5/lstm_6/while/add?
!sequential_5/lstm_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_5/lstm_6/while/add_1/y?
sequential_5/lstm_6/while/add_1AddV2@sequential_5_lstm_6_while_sequential_5_lstm_6_while_loop_counter*sequential_5/lstm_6/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_5/lstm_6/while/add_1?
"sequential_5/lstm_6/while/IdentityIdentity#sequential_5/lstm_6/while/add_1:z:0*
T0*
_output_shapes
: 2$
"sequential_5/lstm_6/while/Identity?
$sequential_5/lstm_6/while/Identity_1IdentityFsequential_5_lstm_6_while_sequential_5_lstm_6_while_maximum_iterations*
T0*
_output_shapes
: 2&
$sequential_5/lstm_6/while/Identity_1?
$sequential_5/lstm_6/while/Identity_2Identity!sequential_5/lstm_6/while/add:z:0*
T0*
_output_shapes
: 2&
$sequential_5/lstm_6/while/Identity_2?
$sequential_5/lstm_6/while/Identity_3IdentityNsequential_5/lstm_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2&
$sequential_5/lstm_6/while/Identity_3?
$sequential_5/lstm_6/while/Identity_4Identity/sequential_5/lstm_6/while/lstm_cell_8/mul_2:z:0*
T0*'
_output_shapes
:?????????2&
$sequential_5/lstm_6/while/Identity_4?
$sequential_5/lstm_6/while/Identity_5Identity/sequential_5/lstm_6/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2&
$sequential_5/lstm_6/while/Identity_5"Q
"sequential_5_lstm_6_while_identity+sequential_5/lstm_6/while/Identity:output:0"U
$sequential_5_lstm_6_while_identity_1-sequential_5/lstm_6/while/Identity_1:output:0"U
$sequential_5_lstm_6_while_identity_2-sequential_5/lstm_6/while/Identity_2:output:0"U
$sequential_5_lstm_6_while_identity_3-sequential_5/lstm_6/while/Identity_3:output:0"U
$sequential_5_lstm_6_while_identity_4-sequential_5/lstm_6/while/Identity_4:output:0"U
$sequential_5_lstm_6_while_identity_5-sequential_5/lstm_6/while/Identity_5:output:0"?
Esequential_5_lstm_6_while_lstm_cell_8_biasadd_readvariableop_resourceGsequential_5_lstm_6_while_lstm_cell_8_biasadd_readvariableop_resource_0"?
Fsequential_5_lstm_6_while_lstm_cell_8_matmul_1_readvariableop_resourceHsequential_5_lstm_6_while_lstm_cell_8_matmul_1_readvariableop_resource_0"?
Dsequential_5_lstm_6_while_lstm_cell_8_matmul_readvariableop_resourceFsequential_5_lstm_6_while_lstm_cell_8_matmul_readvariableop_resource_0"?
=sequential_5_lstm_6_while_sequential_5_lstm_6_strided_slice_1?sequential_5_lstm_6_while_sequential_5_lstm_6_strided_slice_1_0"?
ysequential_5_lstm_6_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_6_tensorarrayunstack_tensorlistfromtensor{sequential_5_lstm_6_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_6_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_200991

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slices
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:????????? 2	
Reshape?
flatten_1/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2009432
flatten_1/PartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape_1/shape/2?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape"flatten_1/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:???????????????????2
	Reshape_1t
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:"?????????????????? :` \
8
_output_shapes&
$:"?????????????????? 
 
_user_specified_nameinputs
?9
?
while_body_205530
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_8_matmul_readvariableop_resource_08
4while_lstm_cell_8_matmul_1_readvariableop_resource_07
3while_lstm_cell_8_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_8_matmul_readvariableop_resource6
2while_lstm_cell_8_matmul_1_readvariableop_resource5
1while_lstm_cell_8_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp?
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/MatMul?
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp?
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/MatMul_1?
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/add?
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp?
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/BiasAddt
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_8/Const?
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_8/split?
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid?
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid_1?
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul?
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Relu?
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul_1?
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/add_1?
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid_2?
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Relu_1?
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_lstm_cell_8_layer_call_fn_205917

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_2017312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????2:?????????:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_201012

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slices
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:????????? 2	
Reshape?
flatten_1/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2009432
flatten_1/PartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape_1/shape/2?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape"flatten_1/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:???????????????????2
	Reshape_1t
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:"?????????????????? :` \
8
_output_shapes&
$:"?????????????????? 
 
_user_specified_nameinputs
?W
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_204478
inputs_0.
*lstm_cell_7_matmul_readvariableop_resource0
,lstm_cell_7_matmul_1_readvariableop_resource/
+lstm_cell_7_biasadd_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!lstm_cell_7/MatMul/ReadVariableOp?
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/MatMul?
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_7/MatMul_1/ReadVariableOp?
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/MatMul_1?
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/add?
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_7/BiasAdd/ReadVariableOp?
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/BiasAddh
lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/Const|
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/split/split_dim?
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_7/split?
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid?
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid_1?
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mulz
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Relu?
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mul_1?
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/add_1?
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid_2y
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Relu_1?
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_204393*
condR
while_cond_204392*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
'__inference_lstm_6_layer_call_fn_205298

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_6_layer_call_and_return_conditional_losses_2028022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????2
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_1_layer_call_fn_204222

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"?????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_2007942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*8
_output_shapes&
$:"?????????????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:"??????????????????	@::22
StatefulPartitionedCallStatefulPartitionedCall:` \
8
_output_shapes&
$:"??????????????????	@
 
_user_specified_nameinputs
?W
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_202620

inputs.
*lstm_cell_7_matmul_readvariableop_resource0
,lstm_cell_7_matmul_1_readvariableop_resource/
+lstm_cell_7_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!lstm_cell_7/MatMul/ReadVariableOp?
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/MatMul?
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_7/MatMul_1/ReadVariableOp?
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/MatMul_1?
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/add?
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_7/BiasAdd/ReadVariableOp?
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/BiasAddh
lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/Const|
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/split/split_dim?
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_7/split?
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid?
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid_1?
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mulz
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Relu?
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mul_1?
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/add_1?
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid_2y
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Relu_1?
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_202535*
condR
while_cond_202534*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?R
?
%sequential_5_lstm_5_while_body_200329D
@sequential_5_lstm_5_while_sequential_5_lstm_5_while_loop_counterJ
Fsequential_5_lstm_5_while_sequential_5_lstm_5_while_maximum_iterations)
%sequential_5_lstm_5_while_placeholder+
'sequential_5_lstm_5_while_placeholder_1+
'sequential_5_lstm_5_while_placeholder_2+
'sequential_5_lstm_5_while_placeholder_3C
?sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1_0
{sequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor_0J
Fsequential_5_lstm_5_while_lstm_cell_7_matmul_readvariableop_resource_0L
Hsequential_5_lstm_5_while_lstm_cell_7_matmul_1_readvariableop_resource_0K
Gsequential_5_lstm_5_while_lstm_cell_7_biasadd_readvariableop_resource_0&
"sequential_5_lstm_5_while_identity(
$sequential_5_lstm_5_while_identity_1(
$sequential_5_lstm_5_while_identity_2(
$sequential_5_lstm_5_while_identity_3(
$sequential_5_lstm_5_while_identity_4(
$sequential_5_lstm_5_while_identity_5A
=sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1}
ysequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensorH
Dsequential_5_lstm_5_while_lstm_cell_7_matmul_readvariableop_resourceJ
Fsequential_5_lstm_5_while_lstm_cell_7_matmul_1_readvariableop_resourceI
Esequential_5_lstm_5_while_lstm_cell_7_biasadd_readvariableop_resource??
Ksequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2M
Ksequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=sequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor_0%sequential_5_lstm_5_while_placeholderTsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02?
=sequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem?
;sequential_5/lstm_5/while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOpFsequential_5_lstm_5_while_lstm_cell_7_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02=
;sequential_5/lstm_5/while/lstm_cell_7/MatMul/ReadVariableOp?
,sequential_5/lstm_5/while/lstm_cell_7/MatMulMatMulDsequential_5/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_5/lstm_5/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,sequential_5/lstm_5/while/lstm_cell_7/MatMul?
=sequential_5/lstm_5/while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOpHsequential_5_lstm_5_while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02?
=sequential_5/lstm_5/while/lstm_cell_7/MatMul_1/ReadVariableOp?
.sequential_5/lstm_5/while/lstm_cell_7/MatMul_1MatMul'sequential_5_lstm_5_while_placeholder_2Esequential_5/lstm_5/while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.sequential_5/lstm_5/while/lstm_cell_7/MatMul_1?
)sequential_5/lstm_5/while/lstm_cell_7/addAddV26sequential_5/lstm_5/while/lstm_cell_7/MatMul:product:08sequential_5/lstm_5/while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2+
)sequential_5/lstm_5/while/lstm_cell_7/add?
<sequential_5/lstm_5/while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOpGsequential_5_lstm_5_while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02>
<sequential_5/lstm_5/while/lstm_cell_7/BiasAdd/ReadVariableOp?
-sequential_5/lstm_5/while/lstm_cell_7/BiasAddBiasAdd-sequential_5/lstm_5/while/lstm_cell_7/add:z:0Dsequential_5/lstm_5/while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-sequential_5/lstm_5/while/lstm_cell_7/BiasAdd?
+sequential_5/lstm_5/while/lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2-
+sequential_5/lstm_5/while/lstm_cell_7/Const?
5sequential_5/lstm_5/while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_5/lstm_5/while/lstm_cell_7/split/split_dim?
+sequential_5/lstm_5/while/lstm_cell_7/splitSplit>sequential_5/lstm_5/while/lstm_cell_7/split/split_dim:output:06sequential_5/lstm_5/while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2-
+sequential_5/lstm_5/while/lstm_cell_7/split?
-sequential_5/lstm_5/while/lstm_cell_7/SigmoidSigmoid4sequential_5/lstm_5/while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22/
-sequential_5/lstm_5/while/lstm_cell_7/Sigmoid?
/sequential_5/lstm_5/while/lstm_cell_7/Sigmoid_1Sigmoid4sequential_5/lstm_5/while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????221
/sequential_5/lstm_5/while/lstm_cell_7/Sigmoid_1?
)sequential_5/lstm_5/while/lstm_cell_7/mulMul3sequential_5/lstm_5/while/lstm_cell_7/Sigmoid_1:y:0'sequential_5_lstm_5_while_placeholder_3*
T0*'
_output_shapes
:?????????22+
)sequential_5/lstm_5/while/lstm_cell_7/mul?
*sequential_5/lstm_5/while/lstm_cell_7/ReluRelu4sequential_5/lstm_5/while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22,
*sequential_5/lstm_5/while/lstm_cell_7/Relu?
+sequential_5/lstm_5/while/lstm_cell_7/mul_1Mul1sequential_5/lstm_5/while/lstm_cell_7/Sigmoid:y:08sequential_5/lstm_5/while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22-
+sequential_5/lstm_5/while/lstm_cell_7/mul_1?
+sequential_5/lstm_5/while/lstm_cell_7/add_1AddV2-sequential_5/lstm_5/while/lstm_cell_7/mul:z:0/sequential_5/lstm_5/while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22-
+sequential_5/lstm_5/while/lstm_cell_7/add_1?
/sequential_5/lstm_5/while/lstm_cell_7/Sigmoid_2Sigmoid4sequential_5/lstm_5/while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????221
/sequential_5/lstm_5/while/lstm_cell_7/Sigmoid_2?
,sequential_5/lstm_5/while/lstm_cell_7/Relu_1Relu/sequential_5/lstm_5/while/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22.
,sequential_5/lstm_5/while/lstm_cell_7/Relu_1?
+sequential_5/lstm_5/while/lstm_cell_7/mul_2Mul3sequential_5/lstm_5/while/lstm_cell_7/Sigmoid_2:y:0:sequential_5/lstm_5/while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22-
+sequential_5/lstm_5/while/lstm_cell_7/mul_2?
>sequential_5/lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_5_lstm_5_while_placeholder_1%sequential_5_lstm_5_while_placeholder/sequential_5/lstm_5/while/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype02@
>sequential_5/lstm_5/while/TensorArrayV2Write/TensorListSetItem?
sequential_5/lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_5/lstm_5/while/add/y?
sequential_5/lstm_5/while/addAddV2%sequential_5_lstm_5_while_placeholder(sequential_5/lstm_5/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_5/lstm_5/while/add?
!sequential_5/lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_5/lstm_5/while/add_1/y?
sequential_5/lstm_5/while/add_1AddV2@sequential_5_lstm_5_while_sequential_5_lstm_5_while_loop_counter*sequential_5/lstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_5/lstm_5/while/add_1?
"sequential_5/lstm_5/while/IdentityIdentity#sequential_5/lstm_5/while/add_1:z:0*
T0*
_output_shapes
: 2$
"sequential_5/lstm_5/while/Identity?
$sequential_5/lstm_5/while/Identity_1IdentityFsequential_5_lstm_5_while_sequential_5_lstm_5_while_maximum_iterations*
T0*
_output_shapes
: 2&
$sequential_5/lstm_5/while/Identity_1?
$sequential_5/lstm_5/while/Identity_2Identity!sequential_5/lstm_5/while/add:z:0*
T0*
_output_shapes
: 2&
$sequential_5/lstm_5/while/Identity_2?
$sequential_5/lstm_5/while/Identity_3IdentityNsequential_5/lstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2&
$sequential_5/lstm_5/while/Identity_3?
$sequential_5/lstm_5/while/Identity_4Identity/sequential_5/lstm_5/while/lstm_cell_7/mul_2:z:0*
T0*'
_output_shapes
:?????????22&
$sequential_5/lstm_5/while/Identity_4?
$sequential_5/lstm_5/while/Identity_5Identity/sequential_5/lstm_5/while/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22&
$sequential_5/lstm_5/while/Identity_5"Q
"sequential_5_lstm_5_while_identity+sequential_5/lstm_5/while/Identity:output:0"U
$sequential_5_lstm_5_while_identity_1-sequential_5/lstm_5/while/Identity_1:output:0"U
$sequential_5_lstm_5_while_identity_2-sequential_5/lstm_5/while/Identity_2:output:0"U
$sequential_5_lstm_5_while_identity_3-sequential_5/lstm_5/while/Identity_3:output:0"U
$sequential_5_lstm_5_while_identity_4-sequential_5/lstm_5/while/Identity_4:output:0"U
$sequential_5_lstm_5_while_identity_5-sequential_5/lstm_5/while/Identity_5:output:0"?
Esequential_5_lstm_5_while_lstm_cell_7_biasadd_readvariableop_resourceGsequential_5_lstm_5_while_lstm_cell_7_biasadd_readvariableop_resource_0"?
Fsequential_5_lstm_5_while_lstm_cell_7_matmul_1_readvariableop_resourceHsequential_5_lstm_5_while_lstm_cell_7_matmul_1_readvariableop_resource_0"?
Dsequential_5_lstm_5_while_lstm_cell_7_matmul_readvariableop_resourceFsequential_5_lstm_5_while_lstm_cell_7_matmul_readvariableop_resource_0"?
=sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1?sequential_5_lstm_5_while_sequential_5_lstm_5_strided_slice_1_0"?
ysequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor{sequential_5_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_dense_12_layer_call_and_return_conditional_losses_202995

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_200840

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_201121

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:??????????:?????????2:?????????2::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates
?$
?
while_body_201547
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_7_201571_0
while_lstm_cell_7_201573_0
while_lstm_cell_7_201575_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_7_201571
while_lstm_cell_7_201573
while_lstm_cell_7_201575??)while/lstm_cell_7/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_7/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_7_201571_0while_lstm_cell_7_201573_0while_lstm_cell_7_201575_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_2011212+
)while/lstm_cell_7/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_7/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_7/StatefulPartitionedCall:output:1*^while/lstm_cell_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_7/StatefulPartitionedCall:output:2*^while/lstm_cell_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_7_201571while_lstm_cell_7_201571_0"6
while_lstm_cell_7_201573while_lstm_cell_7_201573_0"6
while_lstm_cell_7_201575while_lstm_cell_7_201575_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2V
)while/lstm_cell_7/StatefulPartitionedCall)while/lstm_cell_7/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?$
?
while_body_202025
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_8_202049_0
while_lstm_cell_8_202051_0
while_lstm_cell_8_202053_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_8_202049
while_lstm_cell_8_202051
while_lstm_cell_8_202053??)while/lstm_cell_8/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_202049_0while_lstm_cell_8_202051_0while_lstm_cell_8_202053_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_2016982+
)while/lstm_cell_8/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_8/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1*^while/lstm_cell_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2*^while/lstm_cell_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_8_202049while_lstm_cell_8_202049_0"6
while_lstm_cell_8_202051while_lstm_cell_8_202051_0"6
while_lstm_cell_8_202053while_lstm_cell_8_202053_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2V
)while/lstm_cell_8/StatefulPartitionedCall)while/lstm_cell_8/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?9
?
while_body_202382
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_7_matmul_readvariableop_resource_08
4while_lstm_cell_7_matmul_1_readvariableop_resource_07
3while_lstm_cell_7_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_7_matmul_readvariableop_resource6
2while_lstm_cell_7_matmul_1_readvariableop_resource5
1while_lstm_cell_7_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/lstm_cell_7/MatMul/ReadVariableOp?
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/MatMul?
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_7/MatMul_1/ReadVariableOp?
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/MatMul_1?
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/add?
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_7/BiasAdd/ReadVariableOp?
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/BiasAddt
while/lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_7/Const?
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_7/split/split_dim?
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_7/split?
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid?
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid_1?
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul?
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Relu?
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul_1?
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/add_1?
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid_2?
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Relu_1?
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?W
?
B__inference_lstm_6_layer_call_and_return_conditional_losses_205462
inputs_0.
*lstm_cell_8_matmul_readvariableop_resource0
,lstm_cell_8_matmul_1_readvariableop_resource/
+lstm_cell_8_biasadd_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOp?
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/MatMul?
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp?
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/MatMul_1?
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/add?
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/BiasAddh
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/Const|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dim?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_8/split?
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid?
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid_1?
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Relu?
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mul_1?
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/add_1?
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Relu_1?
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_205377*
condR
while_cond_205376*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????2
"
_user_specified_name
inputs/0
?
O
3__inference_time_distributed_3_layer_call_fn_204320

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_2009912
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:"?????????????????? :` \
8
_output_shapes&
$:"?????????????????? 
 
_user_specified_nameinputs
?W
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_204631
inputs_0.
*lstm_cell_7_matmul_readvariableop_resource0
,lstm_cell_7_matmul_1_readvariableop_resource/
+lstm_cell_7_biasadd_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_7/MatMul/ReadVariableOpReadVariableOp*lstm_cell_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!lstm_cell_7/MatMul/ReadVariableOp?
lstm_cell_7/MatMulMatMulstrided_slice_2:output:0)lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/MatMul?
#lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_7/MatMul_1/ReadVariableOp?
lstm_cell_7/MatMul_1MatMulzeros:output:0+lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/MatMul_1?
lstm_cell_7/addAddV2lstm_cell_7/MatMul:product:0lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/add?
"lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_7/BiasAdd/ReadVariableOp?
lstm_cell_7/BiasAddBiasAddlstm_cell_7/add:z:0*lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_7/BiasAddh
lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/Const|
lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_7/split/split_dim?
lstm_cell_7/splitSplit$lstm_cell_7/split/split_dim:output:0lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_7/split?
lstm_cell_7/SigmoidSigmoidlstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid?
lstm_cell_7/Sigmoid_1Sigmoidlstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid_1?
lstm_cell_7/mulMullstm_cell_7/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mulz
lstm_cell_7/ReluRelulstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Relu?
lstm_cell_7/mul_1Mullstm_cell_7/Sigmoid:y:0lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mul_1?
lstm_cell_7/add_1AddV2lstm_cell_7/mul:z:0lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/add_1?
lstm_cell_7/Sigmoid_2Sigmoidlstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Sigmoid_2y
lstm_cell_7/Relu_1Relulstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/Relu_1?
lstm_cell_7/mul_2Mullstm_cell_7/Sigmoid_2:y:0 lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_7/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_7_matmul_readvariableop_resource,lstm_cell_7_matmul_1_readvariableop_resource+lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_204546*
condR
while_cond_204545*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?9
?
while_body_202535
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_7_matmul_readvariableop_resource_08
4while_lstm_cell_7_matmul_1_readvariableop_resource_07
3while_lstm_cell_7_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_7_matmul_readvariableop_resource6
2while_lstm_cell_7_matmul_1_readvariableop_resource5
1while_lstm_cell_7_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/lstm_cell_7/MatMul/ReadVariableOp?
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/MatMul?
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_7/MatMul_1/ReadVariableOp?
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/MatMul_1?
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/add?
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_7/BiasAdd/ReadVariableOp?
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/BiasAddt
while/lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_7/Const?
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_7/split/split_dim?
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_7/split?
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid?
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid_1?
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul?
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Relu?
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul_1?
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/add_1?
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid_2?
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Relu_1?
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
O
3__inference_time_distributed_2_layer_call_fn_204281

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_2009232
PartitionedCall}
IdentityIdentityPartitionedCall:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:"?????????????????? :` \
8
_output_shapes&
$:"?????????????????? 
 
_user_specified_nameinputs
?9
?
while_body_205202
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_8_matmul_readvariableop_resource_08
4while_lstm_cell_8_matmul_1_readvariableop_resource_07
3while_lstm_cell_8_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_8_matmul_readvariableop_resource6
2while_lstm_cell_8_matmul_1_readvariableop_resource5
1while_lstm_cell_8_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp?
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/MatMul?
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp?
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/MatMul_1?
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/add?
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp?
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/BiasAddt
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_8/Const?
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_8/split?
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid?
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid_1?
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul?
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Relu?
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul_1?
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/add_1?
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid_2?
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Relu_1?
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_202381
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_202381___redundant_placeholder04
0while_while_cond_202381___redundant_placeholder14
0while_while_cond_202381___redundant_placeholder24
0while_while_cond_202381___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?a
?
__inference__traced_save_206075
file_prefix.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop6
2savev2_time_distributed_kernel_read_readvariableop4
0savev2_time_distributed_bias_read_readvariableop8
4savev2_time_distributed_1_kernel_read_readvariableop6
2savev2_time_distributed_1_bias_read_readvariableop8
4savev2_lstm_5_lstm_cell_7_kernel_read_readvariableopB
>savev2_lstm_5_lstm_cell_7_recurrent_kernel_read_readvariableop6
2savev2_lstm_5_lstm_cell_7_bias_read_readvariableop8
4savev2_lstm_6_lstm_cell_8_kernel_read_readvariableopB
>savev2_lstm_6_lstm_cell_8_recurrent_kernel_read_readvariableop6
2savev2_lstm_6_lstm_cell_8_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop=
9savev2_adam_time_distributed_kernel_m_read_readvariableop;
7savev2_adam_time_distributed_bias_m_read_readvariableop?
;savev2_adam_time_distributed_1_kernel_m_read_readvariableop=
9savev2_adam_time_distributed_1_bias_m_read_readvariableop?
;savev2_adam_lstm_5_lstm_cell_7_kernel_m_read_readvariableopI
Esavev2_adam_lstm_5_lstm_cell_7_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_5_lstm_cell_7_bias_m_read_readvariableop?
;savev2_adam_lstm_6_lstm_cell_8_kernel_m_read_readvariableopI
Esavev2_adam_lstm_6_lstm_cell_8_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_6_lstm_cell_8_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop=
9savev2_adam_time_distributed_kernel_v_read_readvariableop;
7savev2_adam_time_distributed_bias_v_read_readvariableop?
;savev2_adam_time_distributed_1_kernel_v_read_readvariableop=
9savev2_adam_time_distributed_1_bias_v_read_readvariableop?
;savev2_adam_lstm_5_lstm_cell_7_kernel_v_read_readvariableopI
Esavev2_adam_lstm_5_lstm_cell_7_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_5_lstm_cell_7_bias_v_read_readvariableop?
;savev2_adam_lstm_6_lstm_cell_8_kernel_v_read_readvariableopI
Esavev2_adam_lstm_6_lstm_cell_8_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_6_lstm_cell_8_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b4e85c9005d1484d8ee04ab429925c0c/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_time_distributed_kernel_read_readvariableop0savev2_time_distributed_bias_read_readvariableop4savev2_time_distributed_1_kernel_read_readvariableop2savev2_time_distributed_1_bias_read_readvariableop4savev2_lstm_5_lstm_cell_7_kernel_read_readvariableop>savev2_lstm_5_lstm_cell_7_recurrent_kernel_read_readvariableop2savev2_lstm_5_lstm_cell_7_bias_read_readvariableop4savev2_lstm_6_lstm_cell_8_kernel_read_readvariableop>savev2_lstm_6_lstm_cell_8_recurrent_kernel_read_readvariableop2savev2_lstm_6_lstm_cell_8_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop9savev2_adam_time_distributed_kernel_m_read_readvariableop7savev2_adam_time_distributed_bias_m_read_readvariableop;savev2_adam_time_distributed_1_kernel_m_read_readvariableop9savev2_adam_time_distributed_1_bias_m_read_readvariableop;savev2_adam_lstm_5_lstm_cell_7_kernel_m_read_readvariableopEsavev2_adam_lstm_5_lstm_cell_7_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_5_lstm_cell_7_bias_m_read_readvariableop;savev2_adam_lstm_6_lstm_cell_8_kernel_m_read_readvariableopEsavev2_adam_lstm_6_lstm_cell_8_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_6_lstm_cell_8_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop9savev2_adam_time_distributed_kernel_v_read_readvariableop7savev2_adam_time_distributed_bias_v_read_readvariableop;savev2_adam_time_distributed_1_kernel_v_read_readvariableop9savev2_adam_time_distributed_1_bias_v_read_readvariableop;savev2_adam_lstm_5_lstm_cell_7_kernel_v_read_readvariableopEsavev2_adam_lstm_5_lstm_cell_7_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_5_lstm_cell_7_bias_v_read_readvariableop;savev2_adam_lstm_6_lstm_cell_8_kernel_v_read_readvariableopEsavev2_adam_lstm_6_lstm_cell_8_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_6_lstm_cell_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: : : : : :@:@:@ : :
??:	2?:?:2d:d:d: : : : :::@:@:@ : :
??:	2?:?:2d:d:d:::@:@:@ : :
??:	2?:?:2d:d:d: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:@: 	

_output_shapes
:@:(
$
"
_output_shapes
:@ : 

_output_shapes
: :&"
 
_output_shapes
:
??:%!

_output_shapes
:	2?:!

_output_shapes	
:?:$ 

_output_shapes

:2d:$ 

_output_shapes

:d: 

_output_shapes
:d:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::($
"
_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@ : 

_output_shapes
: :&"
 
_output_shapes
:
??:%!

_output_shapes
:	2?:!

_output_shapes	
:?:$ 

_output_shapes

:2d:$  

_output_shapes

:d: !

_output_shapes
:d:$" 

_output_shapes

:: #

_output_shapes
::($$
"
_output_shapes
:@: %

_output_shapes
:@:(&$
"
_output_shapes
:@ : '

_output_shapes
: :&("
 
_output_shapes
:
??:%)!

_output_shapes
:	2?:!*

_output_shapes	
:?:$+ 

_output_shapes

:2d:$, 

_output_shapes

:d: -

_output_shapes
:d:.

_output_shapes
: 
?
?
%sequential_5_lstm_6_while_cond_200477D
@sequential_5_lstm_6_while_sequential_5_lstm_6_while_loop_counterJ
Fsequential_5_lstm_6_while_sequential_5_lstm_6_while_maximum_iterations)
%sequential_5_lstm_6_while_placeholder+
'sequential_5_lstm_6_while_placeholder_1+
'sequential_5_lstm_6_while_placeholder_2+
'sequential_5_lstm_6_while_placeholder_3F
Bsequential_5_lstm_6_while_less_sequential_5_lstm_6_strided_slice_1\
Xsequential_5_lstm_6_while_sequential_5_lstm_6_while_cond_200477___redundant_placeholder0\
Xsequential_5_lstm_6_while_sequential_5_lstm_6_while_cond_200477___redundant_placeholder1\
Xsequential_5_lstm_6_while_sequential_5_lstm_6_while_cond_200477___redundant_placeholder2\
Xsequential_5_lstm_6_while_sequential_5_lstm_6_while_cond_200477___redundant_placeholder3&
"sequential_5_lstm_6_while_identity
?
sequential_5/lstm_6/while/LessLess%sequential_5_lstm_6_while_placeholderBsequential_5_lstm_6_while_less_sequential_5_lstm_6_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_5/lstm_6/while/Less?
"sequential_5/lstm_6/while/IdentityIdentity"sequential_5/lstm_6/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_5/lstm_6/while/Identity"Q
"sequential_5_lstm_6_while_identity+sequential_5/lstm_6/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?W
?
B__inference_lstm_6_layer_call_and_return_conditional_losses_205615
inputs_0.
*lstm_cell_8_matmul_readvariableop_resource0
,lstm_cell_8_matmul_1_readvariableop_resource/
+lstm_cell_8_biasadd_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOp?
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/MatMul?
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp?
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/MatMul_1?
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/add?
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/BiasAddh
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/Const|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dim?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_8/split?
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid?
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid_1?
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Relu?
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mul_1?
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/add_1?
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Relu_1?
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_205530*
condR
while_cond_205529*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????2
"
_user_specified_name
inputs/0
?B
?
lstm_6_while_body_203934*
&lstm_6_while_lstm_6_while_loop_counter0
,lstm_6_while_lstm_6_while_maximum_iterations
lstm_6_while_placeholder
lstm_6_while_placeholder_1
lstm_6_while_placeholder_2
lstm_6_while_placeholder_3)
%lstm_6_while_lstm_6_strided_slice_1_0e
alstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_6_while_lstm_cell_8_matmul_readvariableop_resource_0?
;lstm_6_while_lstm_cell_8_matmul_1_readvariableop_resource_0>
:lstm_6_while_lstm_cell_8_biasadd_readvariableop_resource_0
lstm_6_while_identity
lstm_6_while_identity_1
lstm_6_while_identity_2
lstm_6_while_identity_3
lstm_6_while_identity_4
lstm_6_while_identity_5'
#lstm_6_while_lstm_6_strided_slice_1c
_lstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensor;
7lstm_6_while_lstm_cell_8_matmul_readvariableop_resource=
9lstm_6_while_lstm_cell_8_matmul_1_readvariableop_resource<
8lstm_6_while_lstm_cell_8_biasadd_readvariableop_resource??
>lstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2@
>lstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensor_0lstm_6_while_placeholderGlstm_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype022
0lstm_6/while/TensorArrayV2Read/TensorListGetItem?
.lstm_6/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp9lstm_6_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype020
.lstm_6/while/lstm_cell_8/MatMul/ReadVariableOp?
lstm_6/while/lstm_cell_8/MatMulMatMul7lstm_6/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_6/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
lstm_6/while/lstm_cell_8/MatMul?
0lstm_6/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp;lstm_6_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype022
0lstm_6/while/lstm_cell_8/MatMul_1/ReadVariableOp?
!lstm_6/while/lstm_cell_8/MatMul_1MatMullstm_6_while_placeholder_28lstm_6/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2#
!lstm_6/while/lstm_cell_8/MatMul_1?
lstm_6/while/lstm_cell_8/addAddV2)lstm_6/while/lstm_cell_8/MatMul:product:0+lstm_6/while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_6/while/lstm_cell_8/add?
/lstm_6/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp:lstm_6_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype021
/lstm_6/while/lstm_cell_8/BiasAdd/ReadVariableOp?
 lstm_6/while/lstm_cell_8/BiasAddBiasAdd lstm_6/while/lstm_cell_8/add:z:07lstm_6/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2"
 lstm_6/while/lstm_cell_8/BiasAdd?
lstm_6/while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_6/while/lstm_cell_8/Const?
(lstm_6/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_6/while/lstm_cell_8/split/split_dim?
lstm_6/while/lstm_cell_8/splitSplit1lstm_6/while/lstm_cell_8/split/split_dim:output:0)lstm_6/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2 
lstm_6/while/lstm_cell_8/split?
 lstm_6/while/lstm_cell_8/SigmoidSigmoid'lstm_6/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2"
 lstm_6/while/lstm_cell_8/Sigmoid?
"lstm_6/while/lstm_cell_8/Sigmoid_1Sigmoid'lstm_6/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????2$
"lstm_6/while/lstm_cell_8/Sigmoid_1?
lstm_6/while/lstm_cell_8/mulMul&lstm_6/while/lstm_cell_8/Sigmoid_1:y:0lstm_6_while_placeholder_3*
T0*'
_output_shapes
:?????????2
lstm_6/while/lstm_cell_8/mul?
lstm_6/while/lstm_cell_8/ReluRelu'lstm_6/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_6/while/lstm_cell_8/Relu?
lstm_6/while/lstm_cell_8/mul_1Mul$lstm_6/while/lstm_cell_8/Sigmoid:y:0+lstm_6/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2 
lstm_6/while/lstm_cell_8/mul_1?
lstm_6/while/lstm_cell_8/add_1AddV2 lstm_6/while/lstm_cell_8/mul:z:0"lstm_6/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2 
lstm_6/while/lstm_cell_8/add_1?
"lstm_6/while/lstm_cell_8/Sigmoid_2Sigmoid'lstm_6/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????2$
"lstm_6/while/lstm_cell_8/Sigmoid_2?
lstm_6/while/lstm_cell_8/Relu_1Relu"lstm_6/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2!
lstm_6/while/lstm_cell_8/Relu_1?
lstm_6/while/lstm_cell_8/mul_2Mul&lstm_6/while/lstm_cell_8/Sigmoid_2:y:0-lstm_6/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2 
lstm_6/while/lstm_cell_8/mul_2?
1lstm_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_6_while_placeholder_1lstm_6_while_placeholder"lstm_6/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_6/while/TensorArrayV2Write/TensorListSetItemj
lstm_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/while/add/y?
lstm_6/while/addAddV2lstm_6_while_placeholderlstm_6/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_6/while/addn
lstm_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/while/add_1/y?
lstm_6/while/add_1AddV2&lstm_6_while_lstm_6_while_loop_counterlstm_6/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_6/while/add_1s
lstm_6/while/IdentityIdentitylstm_6/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_6/while/Identity?
lstm_6/while/Identity_1Identity,lstm_6_while_lstm_6_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_6/while/Identity_1u
lstm_6/while/Identity_2Identitylstm_6/while/add:z:0*
T0*
_output_shapes
: 2
lstm_6/while/Identity_2?
lstm_6/while/Identity_3IdentityAlstm_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_6/while/Identity_3?
lstm_6/while/Identity_4Identity"lstm_6/while/lstm_cell_8/mul_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_6/while/Identity_4?
lstm_6/while/Identity_5Identity"lstm_6/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_6/while/Identity_5"7
lstm_6_while_identitylstm_6/while/Identity:output:0";
lstm_6_while_identity_1 lstm_6/while/Identity_1:output:0";
lstm_6_while_identity_2 lstm_6/while/Identity_2:output:0";
lstm_6_while_identity_3 lstm_6/while/Identity_3:output:0";
lstm_6_while_identity_4 lstm_6/while/Identity_4:output:0";
lstm_6_while_identity_5 lstm_6/while/Identity_5:output:0"L
#lstm_6_while_lstm_6_strided_slice_1%lstm_6_while_lstm_6_strided_slice_1_0"v
8lstm_6_while_lstm_cell_8_biasadd_readvariableop_resource:lstm_6_while_lstm_cell_8_biasadd_readvariableop_resource_0"x
9lstm_6_while_lstm_cell_8_matmul_1_readvariableop_resource;lstm_6_while_lstm_cell_8_matmul_1_readvariableop_resource_0"t
7lstm_6_while_lstm_cell_8_matmul_readvariableop_resource9lstm_6_while_lstm_cell_8_matmul_readvariableop_resource_0"?
_lstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensoralstm_6_while_tensorarrayv2read_tensorlistgetitem_lstm_6_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_200923

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slices
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:????????? 2	
Reshape?
max_pooling1d_1/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_2008402!
max_pooling1d_1/PartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/3?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape(max_pooling1d_1/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2
	Reshape_1w
IdentityIdentityReshape_1:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:"?????????????????? :` \
8
_output_shapes&
$:"?????????????????? 
 
_user_specified_nameinputs
?B
?
lstm_5_while_body_203392*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3)
%lstm_5_while_lstm_5_strided_slice_1_0e
alstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_5_while_lstm_cell_7_matmul_readvariableop_resource_0?
;lstm_5_while_lstm_cell_7_matmul_1_readvariableop_resource_0>
:lstm_5_while_lstm_cell_7_biasadd_readvariableop_resource_0
lstm_5_while_identity
lstm_5_while_identity_1
lstm_5_while_identity_2
lstm_5_while_identity_3
lstm_5_while_identity_4
lstm_5_while_identity_5'
#lstm_5_while_lstm_5_strided_slice_1c
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor;
7lstm_5_while_lstm_cell_7_matmul_readvariableop_resource=
9lstm_5_while_lstm_cell_7_matmul_1_readvariableop_resource<
8lstm_5_while_lstm_cell_7_biasadd_readvariableop_resource??
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2@
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0lstm_5_while_placeholderGlstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype022
0lstm_5/while/TensorArrayV2Read/TensorListGetItem?
.lstm_5/while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp9lstm_5_while_lstm_cell_7_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype020
.lstm_5/while/lstm_cell_7/MatMul/ReadVariableOp?
lstm_5/while/lstm_cell_7/MatMulMatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_5/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
lstm_5/while/lstm_cell_7/MatMul?
0lstm_5/while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp;lstm_5_while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype022
0lstm_5/while/lstm_cell_7/MatMul_1/ReadVariableOp?
!lstm_5/while/lstm_cell_7/MatMul_1MatMullstm_5_while_placeholder_28lstm_5/while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_5/while/lstm_cell_7/MatMul_1?
lstm_5/while/lstm_cell_7/addAddV2)lstm_5/while/lstm_cell_7/MatMul:product:0+lstm_5/while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_5/while/lstm_cell_7/add?
/lstm_5/while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp:lstm_5_while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/lstm_5/while/lstm_cell_7/BiasAdd/ReadVariableOp?
 lstm_5/while/lstm_cell_7/BiasAddBiasAdd lstm_5/while/lstm_cell_7/add:z:07lstm_5/while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 lstm_5/while/lstm_cell_7/BiasAdd?
lstm_5/while/lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_5/while/lstm_cell_7/Const?
(lstm_5/while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_5/while/lstm_cell_7/split/split_dim?
lstm_5/while/lstm_cell_7/splitSplit1lstm_5/while/lstm_cell_7/split/split_dim:output:0)lstm_5/while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2 
lstm_5/while/lstm_cell_7/split?
 lstm_5/while/lstm_cell_7/SigmoidSigmoid'lstm_5/while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22"
 lstm_5/while/lstm_cell_7/Sigmoid?
"lstm_5/while/lstm_cell_7/Sigmoid_1Sigmoid'lstm_5/while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????22$
"lstm_5/while/lstm_cell_7/Sigmoid_1?
lstm_5/while/lstm_cell_7/mulMul&lstm_5/while/lstm_cell_7/Sigmoid_1:y:0lstm_5_while_placeholder_3*
T0*'
_output_shapes
:?????????22
lstm_5/while/lstm_cell_7/mul?
lstm_5/while/lstm_cell_7/ReluRelu'lstm_5/while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_5/while/lstm_cell_7/Relu?
lstm_5/while/lstm_cell_7/mul_1Mul$lstm_5/while/lstm_cell_7/Sigmoid:y:0+lstm_5/while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22 
lstm_5/while/lstm_cell_7/mul_1?
lstm_5/while/lstm_cell_7/add_1AddV2 lstm_5/while/lstm_cell_7/mul:z:0"lstm_5/while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22 
lstm_5/while/lstm_cell_7/add_1?
"lstm_5/while/lstm_cell_7/Sigmoid_2Sigmoid'lstm_5/while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????22$
"lstm_5/while/lstm_cell_7/Sigmoid_2?
lstm_5/while/lstm_cell_7/Relu_1Relu"lstm_5/while/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22!
lstm_5/while/lstm_cell_7/Relu_1?
lstm_5/while/lstm_cell_7/mul_2Mul&lstm_5/while/lstm_cell_7/Sigmoid_2:y:0-lstm_5/while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22 
lstm_5/while/lstm_cell_7/mul_2?
1lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_5_while_placeholder_1lstm_5_while_placeholder"lstm_5/while/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_5/while/TensorArrayV2Write/TensorListSetItemj
lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add/y?
lstm_5/while/addAddV2lstm_5_while_placeholderlstm_5/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/addn
lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add_1/y?
lstm_5/while/add_1AddV2&lstm_5_while_lstm_5_while_loop_counterlstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/add_1s
lstm_5/while/IdentityIdentitylstm_5/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_5/while/Identity?
lstm_5/while/Identity_1Identity,lstm_5_while_lstm_5_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_5/while/Identity_1u
lstm_5/while/Identity_2Identitylstm_5/while/add:z:0*
T0*
_output_shapes
: 2
lstm_5/while/Identity_2?
lstm_5/while/Identity_3IdentityAlstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_5/while/Identity_3?
lstm_5/while/Identity_4Identity"lstm_5/while/lstm_cell_7/mul_2:z:0*
T0*'
_output_shapes
:?????????22
lstm_5/while/Identity_4?
lstm_5/while/Identity_5Identity"lstm_5/while/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_5/while/Identity_5"7
lstm_5_while_identitylstm_5/while/Identity:output:0";
lstm_5_while_identity_1 lstm_5/while/Identity_1:output:0";
lstm_5_while_identity_2 lstm_5/while/Identity_2:output:0";
lstm_5_while_identity_3 lstm_5/while/Identity_3:output:0";
lstm_5_while_identity_4 lstm_5/while/Identity_4:output:0";
lstm_5_while_identity_5 lstm_5/while/Identity_5:output:0"L
#lstm_5_while_lstm_5_strided_slice_1%lstm_5_while_lstm_5_strided_slice_1_0"v
8lstm_5_while_lstm_cell_7_biasadd_readvariableop_resource:lstm_5_while_lstm_cell_7_biasadd_readvariableop_resource_0"x
9lstm_5_while_lstm_cell_7_matmul_1_readvariableop_resource;lstm_5_while_lstm_cell_7_matmul_1_readvariableop_resource_0"t
7lstm_5_while_lstm_cell_7_matmul_readvariableop_resource9lstm_5_while_lstm_cell_7_matmul_readvariableop_resource_0"?
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensoralstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_202869
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_202869___redundant_placeholder04
0while_while_cond_202869___redundant_placeholder14
0while_while_cond_202869___redundant_placeholder24
0while_while_cond_202869___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?	
?
-__inference_sequential_5_layer_call_fn_203200
time_distributed_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltime_distributed_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_2031732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:p l
8
_output_shapes&
$:"??????????????????

0
_user_specified_nametime_distributed_input
?	
?
-__inference_sequential_5_layer_call_fn_203128
time_distributed_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltime_distributed_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_2031012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:p l
8
_output_shapes&
$:"??????????????????

0
_user_specified_nametime_distributed_input
?
?
1__inference_time_distributed_layer_call_fn_204148

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"??????????????????	@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_2006632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*8
_output_shapes&
$:"??????????????????	@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:"??????????????????
::22
StatefulPartitionedCallStatefulPartitionedCall:` \
8
_output_shapes&
$:"??????????????????

 
_user_specified_nameinputs
?
?
while_cond_204873
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_204873___redundant_placeholder04
0while_while_cond_204873___redundant_placeholder14
0while_while_cond_204873___redundant_placeholder24
0while_while_cond_204873___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?	
?
-__inference_sequential_5_layer_call_fn_204083

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_2031732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
8
_output_shapes&
$:"??????????????????

 
_user_specified_nameinputs
?
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_204271

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slices
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:????????? 2	
Reshape?
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_1/ExpandDims/dim?
max_pooling1d_1/ExpandDims
ExpandDimsReshape:output:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2
max_pooling1d_1/ExpandDims?
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling1d_1/MaxPool?
max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims
2
max_pooling1d_1/Squeezeq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/3?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape max_pooling1d_1/Squeeze:output:0Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2
	Reshape_1w
IdentityIdentityReshape_1:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:"?????????????????? :` \
8
_output_shapes&
$:"?????????????????? 
 
_user_specified_nameinputs
?
?
D__inference_dense_12_layer_call_and_return_conditional_losses_205647

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
lstm_6_while_cond_203540*
&lstm_6_while_lstm_6_while_loop_counter0
,lstm_6_while_lstm_6_while_maximum_iterations
lstm_6_while_placeholder
lstm_6_while_placeholder_1
lstm_6_while_placeholder_2
lstm_6_while_placeholder_3,
(lstm_6_while_less_lstm_6_strided_slice_1B
>lstm_6_while_lstm_6_while_cond_203540___redundant_placeholder0B
>lstm_6_while_lstm_6_while_cond_203540___redundant_placeholder1B
>lstm_6_while_lstm_6_while_cond_203540___redundant_placeholder2B
>lstm_6_while_lstm_6_while_cond_203540___redundant_placeholder3
lstm_6_while_identity
?
lstm_6/while/LessLesslstm_6_while_placeholder(lstm_6_while_less_lstm_6_strided_slice_1*
T0*
_output_shapes
: 2
lstm_6/while/Lessr
lstm_6/while/IdentityIdentitylstm_6/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_6/while/Identity"7
lstm_6_while_identitylstm_6/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?9
?
while_body_204874
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_7_matmul_readvariableop_resource_08
4while_lstm_cell_7_matmul_1_readvariableop_resource_07
3while_lstm_cell_7_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_7_matmul_readvariableop_resource6
2while_lstm_cell_7_matmul_1_readvariableop_resource5
1while_lstm_cell_7_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/lstm_cell_7/MatMul/ReadVariableOp?
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/MatMul?
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_7/MatMul_1/ReadVariableOp?
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/MatMul_1?
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/add?
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_7/BiasAdd/ReadVariableOp?
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/BiasAddt
while/lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_7/Const?
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_7/split/split_dim?
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_7/split?
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid?
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid_1?
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul?
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Relu?
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul_1?
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/add_1?
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid_2?
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Relu_1?
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?$
?
while_body_201415
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_7_201439_0
while_lstm_cell_7_201441_0
while_lstm_cell_7_201443_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_7_201439
while_lstm_cell_7_201441
while_lstm_cell_7_201443??)while/lstm_cell_7/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_7/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_7_201439_0while_lstm_cell_7_201441_0while_lstm_cell_7_201443_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_2010882+
)while/lstm_cell_7/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_7/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_7/StatefulPartitionedCall:output:1*^while/lstm_cell_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_7/StatefulPartitionedCall:output:2*^while/lstm_cell_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_7_201439while_lstm_cell_7_201439_0"6
while_lstm_cell_7_201441while_lstm_cell_7_201441_0"6
while_lstm_cell_7_201443while_lstm_cell_7_201443_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2V
)while/lstm_cell_7/StatefulPartitionedCall)while/lstm_cell_7/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_conv1d_2_layer_call_and_return_conditional_losses_205672

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????	@*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????	@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????	@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????	@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????	@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????
:::S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?
~
)__inference_dense_12_layer_call_fn_205656

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_2029952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_200693

inputs
conv1d_2_200682
conv1d_2_200684
identity?? conv1d_2/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slices
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????
      2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????
2	
Reshape?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv1d_2_200682conv1d_2_200684*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????	@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_2005962"
 conv1d_2/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape_1/shape/3?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape)conv1d_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????	@2
	Reshape_1?
IdentityIdentityReshape_1:output:0!^conv1d_2/StatefulPartitionedCall*
T0*8
_output_shapes&
$:"??????????????????	@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:"??????????????????
::2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall:` \
8
_output_shapes&
$:"??????????????????

 
_user_specified_nameinputs
?9
?
while_body_204393
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_7_matmul_readvariableop_resource_08
4while_lstm_cell_7_matmul_1_readvariableop_resource_07
3while_lstm_cell_7_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_7_matmul_readvariableop_resource6
2while_lstm_cell_7_matmul_1_readvariableop_resource5
1while_lstm_cell_7_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_7_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/lstm_cell_7/MatMul/ReadVariableOp?
while/lstm_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/MatMul?
)while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_7/MatMul_1/ReadVariableOp?
while/lstm_cell_7/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/MatMul_1?
while/lstm_cell_7/addAddV2"while/lstm_cell_7/MatMul:product:0$while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/add?
(while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_7/BiasAdd/ReadVariableOp?
while/lstm_cell_7/BiasAddBiasAddwhile/lstm_cell_7/add:z:00while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_7/BiasAddt
while/lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_7/Const?
!while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_7/split/split_dim?
while/lstm_cell_7/splitSplit*while/lstm_cell_7/split/split_dim:output:0"while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_7/split?
while/lstm_cell_7/SigmoidSigmoid while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid?
while/lstm_cell_7/Sigmoid_1Sigmoid while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid_1?
while/lstm_cell_7/mulMulwhile/lstm_cell_7/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul?
while/lstm_cell_7/ReluRelu while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Relu?
while/lstm_cell_7/mul_1Mulwhile/lstm_cell_7/Sigmoid:y:0$while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul_1?
while/lstm_cell_7/add_1AddV2while/lstm_cell_7/mul:z:0while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/add_1?
while/lstm_cell_7/Sigmoid_2Sigmoid while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Sigmoid_2?
while/lstm_cell_7/Relu_1Reluwhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/Relu_1?
while/lstm_cell_7/mul_2Mulwhile/lstm_cell_7/Sigmoid_2:y:0&while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_7/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_7/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_7_biasadd_readvariableop_resource3while_lstm_cell_7_biasadd_readvariableop_resource_0"j
2while_lstm_cell_7_matmul_1_readvariableop_resource4while_lstm_cell_7_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_7_matmul_readvariableop_resource2while_lstm_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_205529
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_205529___redundant_placeholder04
0while_while_cond_205529___redundant_placeholder14
0while_while_cond_205529___redundant_placeholder24
0while_while_cond_205529___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
'__inference_lstm_5_layer_call_fn_204653
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_2016162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_201414
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_201414___redundant_placeholder04
0while_while_cond_201414___redundant_placeholder14
0while_while_cond_201414___redundant_placeholder24
0while_while_cond_201414___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?	
?
$__inference_signature_wrapper_203239
time_distributed_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltime_distributed_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_2005692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:p l
8
_output_shapes&
$:"??????????????????

0
_user_specified_nametime_distributed_input
?
O
3__inference_time_distributed_2_layer_call_fn_204276

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"?????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_2009012
PartitionedCall}
IdentityIdentityPartitionedCall:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:"?????????????????? :` \
8
_output_shapes&
$:"?????????????????? 
 
_user_specified_nameinputs
??
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_203632

inputsI
Etime_distributed_conv1d_2_conv1d_expanddims_1_readvariableop_resource=
9time_distributed_conv1d_2_biasadd_readvariableop_resourceK
Gtime_distributed_1_conv1d_3_conv1d_expanddims_1_readvariableop_resource?
;time_distributed_1_conv1d_3_biasadd_readvariableop_resource5
1lstm_5_lstm_cell_7_matmul_readvariableop_resource7
3lstm_5_lstm_cell_7_matmul_1_readvariableop_resource6
2lstm_5_lstm_cell_7_biasadd_readvariableop_resource5
1lstm_6_lstm_cell_8_matmul_readvariableop_resource7
3lstm_6_lstm_cell_8_matmul_1_readvariableop_resource6
2lstm_6_lstm_cell_8_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity??lstm_5/while?lstm_6/whilef
time_distributed/ShapeShapeinputs*
T0*
_output_shapes
:2
time_distributed/Shape?
$time_distributed/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$time_distributed/strided_slice/stack?
&time_distributed/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed/strided_slice/stack_1?
&time_distributed/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed/strided_slice/stack_2?
time_distributed/strided_sliceStridedSlicetime_distributed/Shape:output:0-time_distributed/strided_slice/stack:output:0/time_distributed/strided_slice/stack_1:output:0/time_distributed/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
time_distributed/strided_slice?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????
      2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????
2
time_distributed/Reshape?
/time_distributed/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/time_distributed/conv1d_2/conv1d/ExpandDims/dim?
+time_distributed/conv1d_2/conv1d/ExpandDims
ExpandDims!time_distributed/Reshape:output:08time_distributed/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
2-
+time_distributed/conv1d_2/conv1d/ExpandDims?
<time_distributed/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEtime_distributed_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<time_distributed/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
1time_distributed/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1time_distributed/conv1d_2/conv1d/ExpandDims_1/dim?
-time_distributed/conv1d_2/conv1d/ExpandDims_1
ExpandDimsDtime_distributed/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0:time_distributed/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-time_distributed/conv1d_2/conv1d/ExpandDims_1?
 time_distributed/conv1d_2/conv1dConv2D4time_distributed/conv1d_2/conv1d/ExpandDims:output:06time_distributed/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????	@*
paddingVALID*
strides
2"
 time_distributed/conv1d_2/conv1d?
(time_distributed/conv1d_2/conv1d/SqueezeSqueeze)time_distributed/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????	@*
squeeze_dims

?????????2*
(time_distributed/conv1d_2/conv1d/Squeeze?
0time_distributed/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp9time_distributed_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0time_distributed/conv1d_2/BiasAdd/ReadVariableOp?
!time_distributed/conv1d_2/BiasAddBiasAdd1time_distributed/conv1d_2/conv1d/Squeeze:output:08time_distributed/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????	@2#
!time_distributed/conv1d_2/BiasAdd?
time_distributed/conv1d_2/ReluRelu*time_distributed/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????	@2 
time_distributed/conv1d_2/Relu?
"time_distributed/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"time_distributed/Reshape_1/shape/0?
"time_distributed/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2$
"time_distributed/Reshape_1/shape/2?
"time_distributed/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2$
"time_distributed/Reshape_1/shape/3?
 time_distributed/Reshape_1/shapePack+time_distributed/Reshape_1/shape/0:output:0'time_distributed/strided_slice:output:0+time_distributed/Reshape_1/shape/2:output:0+time_distributed/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 time_distributed/Reshape_1/shape?
time_distributed/Reshape_1Reshape,time_distributed/conv1d_2/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????	@2
time_distributed/Reshape_1?
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????
      2"
 time_distributed/Reshape_2/shape?
time_distributed/Reshape_2Reshapeinputs)time_distributed/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????
2
time_distributed/Reshape_2?
time_distributed_1/ShapeShape#time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:2
time_distributed_1/Shape?
&time_distributed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed_1/strided_slice/stack?
(time_distributed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_1/strided_slice/stack_1?
(time_distributed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_1/strided_slice/stack_2?
 time_distributed_1/strided_sliceStridedSlice!time_distributed_1/Shape:output:0/time_distributed_1/strided_slice/stack:output:01time_distributed_1/strided_slice/stack_1:output:01time_distributed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 time_distributed_1/strided_slice?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????	   @   2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????	@2
time_distributed_1/Reshape?
1time_distributed_1/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1time_distributed_1/conv1d_3/conv1d/ExpandDims/dim?
-time_distributed_1/conv1d_3/conv1d/ExpandDims
ExpandDims#time_distributed_1/Reshape:output:0:time_distributed_1/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????	@2/
-time_distributed_1/conv1d_3/conv1d/ExpandDims?
>time_distributed_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpGtime_distributed_1_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02@
>time_distributed_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
3time_distributed_1/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3time_distributed_1/conv1d_3/conv1d/ExpandDims_1/dim?
/time_distributed_1/conv1d_3/conv1d/ExpandDims_1
ExpandDimsFtime_distributed_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0<time_distributed_1/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 21
/time_distributed_1/conv1d_3/conv1d/ExpandDims_1?
"time_distributed_1/conv1d_3/conv1dConv2D6time_distributed_1/conv1d_3/conv1d/ExpandDims:output:08time_distributed_1/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2$
"time_distributed_1/conv1d_3/conv1d?
*time_distributed_1/conv1d_3/conv1d/SqueezeSqueeze+time_distributed_1/conv1d_3/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2,
*time_distributed_1/conv1d_3/conv1d/Squeeze?
2time_distributed_1/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp;time_distributed_1_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2time_distributed_1/conv1d_3/BiasAdd/ReadVariableOp?
#time_distributed_1/conv1d_3/BiasAddBiasAdd3time_distributed_1/conv1d_3/conv1d/Squeeze:output:0:time_distributed_1/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2%
#time_distributed_1/conv1d_3/BiasAdd?
 time_distributed_1/conv1d_3/ReluRelu,time_distributed_1/conv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2"
 time_distributed_1/conv1d_3/Relu?
$time_distributed_1/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$time_distributed_1/Reshape_1/shape/0?
$time_distributed_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$time_distributed_1/Reshape_1/shape/2?
$time_distributed_1/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2&
$time_distributed_1/Reshape_1/shape/3?
"time_distributed_1/Reshape_1/shapePack-time_distributed_1/Reshape_1/shape/0:output:0)time_distributed_1/strided_slice:output:0-time_distributed_1/Reshape_1/shape/2:output:0-time_distributed_1/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"time_distributed_1/Reshape_1/shape?
time_distributed_1/Reshape_1Reshape.time_distributed_1/conv1d_3/Relu:activations:0+time_distributed_1/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2
time_distributed_1/Reshape_1?
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????	   @   2$
"time_distributed_1/Reshape_2/shape?
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????	@2
time_distributed_1/Reshape_2?
time_distributed_2/ShapeShape%time_distributed_1/Reshape_1:output:0*
T0*
_output_shapes
:2
time_distributed_2/Shape?
&time_distributed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed_2/strided_slice/stack?
(time_distributed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_2/strided_slice/stack_1?
(time_distributed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_2/strided_slice/stack_2?
 time_distributed_2/strided_sliceStridedSlice!time_distributed_2/Shape:output:0/time_distributed_2/strided_slice/stack:output:01time_distributed_2/strided_slice/stack_1:output:01time_distributed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 time_distributed_2/strided_slice?
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2"
 time_distributed_2/Reshape/shape?
time_distributed_2/ReshapeReshape%time_distributed_1/Reshape_1:output:0)time_distributed_2/Reshape/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_2/Reshape?
1time_distributed_2/max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :23
1time_distributed_2/max_pooling1d_1/ExpandDims/dim?
-time_distributed_2/max_pooling1d_1/ExpandDims
ExpandDims#time_distributed_2/Reshape:output:0:time_distributed_2/max_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2/
-time_distributed_2/max_pooling1d_1/ExpandDims?
*time_distributed_2/max_pooling1d_1/MaxPoolMaxPool6time_distributed_2/max_pooling1d_1/ExpandDims:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2,
*time_distributed_2/max_pooling1d_1/MaxPool?
*time_distributed_2/max_pooling1d_1/SqueezeSqueeze3time_distributed_2/max_pooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims
2,
*time_distributed_2/max_pooling1d_1/Squeeze?
$time_distributed_2/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$time_distributed_2/Reshape_1/shape/0?
$time_distributed_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$time_distributed_2/Reshape_1/shape/2?
$time_distributed_2/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2&
$time_distributed_2/Reshape_1/shape/3?
"time_distributed_2/Reshape_1/shapePack-time_distributed_2/Reshape_1/shape/0:output:0)time_distributed_2/strided_slice:output:0-time_distributed_2/Reshape_1/shape/2:output:0-time_distributed_2/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"time_distributed_2/Reshape_1/shape?
time_distributed_2/Reshape_1Reshape3time_distributed_2/max_pooling1d_1/Squeeze:output:0+time_distributed_2/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2
time_distributed_2/Reshape_1?
"time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2$
"time_distributed_2/Reshape_2/shape?
time_distributed_2/Reshape_2Reshape%time_distributed_1/Reshape_1:output:0+time_distributed_2/Reshape_2/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_2/Reshape_2?
time_distributed_3/ShapeShape%time_distributed_2/Reshape_1:output:0*
T0*
_output_shapes
:2
time_distributed_3/Shape?
&time_distributed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed_3/strided_slice/stack?
(time_distributed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_3/strided_slice/stack_1?
(time_distributed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_3/strided_slice/stack_2?
 time_distributed_3/strided_sliceStridedSlice!time_distributed_3/Shape:output:0/time_distributed_3/strided_slice/stack:output:01time_distributed_3/strided_slice/stack_1:output:01time_distributed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 time_distributed_3/strided_slice?
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2"
 time_distributed_3/Reshape/shape?
time_distributed_3/ReshapeReshape%time_distributed_2/Reshape_1:output:0)time_distributed_3/Reshape/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_3/Reshape?
"time_distributed_3/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2$
"time_distributed_3/flatten_1/Const?
$time_distributed_3/flatten_1/ReshapeReshape#time_distributed_3/Reshape:output:0+time_distributed_3/flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2&
$time_distributed_3/flatten_1/Reshape?
$time_distributed_3/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$time_distributed_3/Reshape_1/shape/0?
$time_distributed_3/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2&
$time_distributed_3/Reshape_1/shape/2?
"time_distributed_3/Reshape_1/shapePack-time_distributed_3/Reshape_1/shape/0:output:0)time_distributed_3/strided_slice:output:0-time_distributed_3/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2$
"time_distributed_3/Reshape_1/shape?
time_distributed_3/Reshape_1Reshape-time_distributed_3/flatten_1/Reshape:output:0+time_distributed_3/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:???????????????????2
time_distributed_3/Reshape_1?
"time_distributed_3/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2$
"time_distributed_3/Reshape_2/shape?
time_distributed_3/Reshape_2Reshape%time_distributed_2/Reshape_1:output:0+time_distributed_3/Reshape_2/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_3/Reshape_2q
lstm_5/ShapeShape%time_distributed_3/Reshape_1:output:0*
T0*
_output_shapes
:2
lstm_5/Shape?
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice/stack?
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_1?
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_2?
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slicej
lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros/mul/y?
lstm_5/zeros/mulMullstm_5/strided_slice:output:0lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/mulm
lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros/Less/y?
lstm_5/zeros/LessLesslstm_5/zeros/mul:z:0lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/Lessp
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros/packed/1?
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros/packedm
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros/Const?
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/zerosn
lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros_1/mul/y?
lstm_5/zeros_1/mulMullstm_5/strided_slice:output:0lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/mulq
lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros_1/Less/y?
lstm_5/zeros_1/LessLesslstm_5/zeros_1/mul:z:0lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/Lesst
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros_1/packed/1?
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros_1/packedq
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros_1/Const?
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/zeros_1?
lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose/perm?
lstm_5/transpose	Transpose%time_distributed_3/Reshape_1:output:0lstm_5/transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
lstm_5/transposed
lstm_5/Shape_1Shapelstm_5/transpose:y:0*
T0*
_output_shapes
:2
lstm_5/Shape_1?
lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_1/stack?
lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_1?
lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_2?
lstm_5/strided_slice_1StridedSlicelstm_5/Shape_1:output:0%lstm_5/strided_slice_1/stack:output:0'lstm_5/strided_slice_1/stack_1:output:0'lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slice_1?
"lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_5/TensorArrayV2/element_shape?
lstm_5/TensorArrayV2TensorListReserve+lstm_5/TensorArrayV2/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2?
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2>
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_5/transpose:y:0Elstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_5/TensorArrayUnstack/TensorListFromTensor?
lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_2/stack?
lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_1?
lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_2?
lstm_5/strided_slice_2StridedSlicelstm_5/transpose:y:0%lstm_5/strided_slice_2/stack:output:0'lstm_5/strided_slice_2/stack_1:output:0'lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_5/strided_slice_2?
(lstm_5/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp1lstm_5_lstm_cell_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(lstm_5/lstm_cell_7/MatMul/ReadVariableOp?
lstm_5/lstm_cell_7/MatMulMatMullstm_5/strided_slice_2:output:00lstm_5/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_7/MatMul?
*lstm_5/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp3lstm_5_lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02,
*lstm_5/lstm_cell_7/MatMul_1/ReadVariableOp?
lstm_5/lstm_cell_7/MatMul_1MatMullstm_5/zeros:output:02lstm_5/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_7/MatMul_1?
lstm_5/lstm_cell_7/addAddV2#lstm_5/lstm_cell_7/MatMul:product:0%lstm_5/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_7/add?
)lstm_5/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp2lstm_5_lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)lstm_5/lstm_cell_7/BiasAdd/ReadVariableOp?
lstm_5/lstm_cell_7/BiasAddBiasAddlstm_5/lstm_cell_7/add:z:01lstm_5/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_7/BiasAddv
lstm_5/lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/lstm_cell_7/Const?
"lstm_5/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_5/lstm_cell_7/split/split_dim?
lstm_5/lstm_cell_7/splitSplit+lstm_5/lstm_cell_7/split/split_dim:output:0#lstm_5/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_5/lstm_cell_7/split?
lstm_5/lstm_cell_7/SigmoidSigmoid!lstm_5/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/Sigmoid?
lstm_5/lstm_cell_7/Sigmoid_1Sigmoid!lstm_5/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/Sigmoid_1?
lstm_5/lstm_cell_7/mulMul lstm_5/lstm_cell_7/Sigmoid_1:y:0lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/mul?
lstm_5/lstm_cell_7/ReluRelu!lstm_5/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/Relu?
lstm_5/lstm_cell_7/mul_1Mullstm_5/lstm_cell_7/Sigmoid:y:0%lstm_5/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/mul_1?
lstm_5/lstm_cell_7/add_1AddV2lstm_5/lstm_cell_7/mul:z:0lstm_5/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/add_1?
lstm_5/lstm_cell_7/Sigmoid_2Sigmoid!lstm_5/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/Sigmoid_2?
lstm_5/lstm_cell_7/Relu_1Relulstm_5/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/Relu_1?
lstm_5/lstm_cell_7/mul_2Mul lstm_5/lstm_cell_7/Sigmoid_2:y:0'lstm_5/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/mul_2?
$lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2&
$lstm_5/TensorArrayV2_1/element_shape?
lstm_5/TensorArrayV2_1TensorListReserve-lstm_5/TensorArrayV2_1/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2_1\
lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/time?
lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_5/while/maximum_iterationsx
lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/while/loop_counter?
lstm_5/whileWhile"lstm_5/while/loop_counter:output:0(lstm_5/while/maximum_iterations:output:0lstm_5/time:output:0lstm_5/TensorArrayV2_1:handle:0lstm_5/zeros:output:0lstm_5/zeros_1:output:0lstm_5/strided_slice_1:output:0>lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_5_lstm_cell_7_matmul_readvariableop_resource3lstm_5_lstm_cell_7_matmul_1_readvariableop_resource2lstm_5_lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_5_while_body_203392*$
condR
lstm_5_while_cond_203391*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
lstm_5/while?
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_5/TensorArrayV2Stack/TensorListStackTensorListStacklstm_5/while:output:3@lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02+
)lstm_5/TensorArrayV2Stack/TensorListStack?
lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_5/strided_slice_3/stack?
lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_5/strided_slice_3/stack_1?
lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_3/stack_2?
lstm_5/strided_slice_3StridedSlice2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_5/strided_slice_3/stack:output:0'lstm_5/strided_slice_3/stack_1:output:0'lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_5/strided_slice_3?
lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose_1/perm?
lstm_5/transpose_1	Transpose2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_5/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
lstm_5/transpose_1t
lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/runtimeb
lstm_6/ShapeShapelstm_5/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_6/Shape?
lstm_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_6/strided_slice/stack?
lstm_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_6/strided_slice/stack_1?
lstm_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_6/strided_slice/stack_2?
lstm_6/strided_sliceStridedSlicelstm_6/Shape:output:0#lstm_6/strided_slice/stack:output:0%lstm_6/strided_slice/stack_1:output:0%lstm_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_6/strided_slicej
lstm_6/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/zeros/mul/y?
lstm_6/zeros/mulMullstm_6/strided_slice:output:0lstm_6/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros/mulm
lstm_6/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/zeros/Less/y?
lstm_6/zeros/LessLesslstm_6/zeros/mul:z:0lstm_6/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros/Lessp
lstm_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/zeros/packed/1?
lstm_6/zeros/packedPacklstm_6/strided_slice:output:0lstm_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_6/zeros/packedm
lstm_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_6/zeros/Const?
lstm_6/zerosFilllstm_6/zeros/packed:output:0lstm_6/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_6/zerosn
lstm_6/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/zeros_1/mul/y?
lstm_6/zeros_1/mulMullstm_6/strided_slice:output:0lstm_6/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros_1/mulq
lstm_6/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/zeros_1/Less/y?
lstm_6/zeros_1/LessLesslstm_6/zeros_1/mul:z:0lstm_6/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros_1/Lesst
lstm_6/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/zeros_1/packed/1?
lstm_6/zeros_1/packedPacklstm_6/strided_slice:output:0 lstm_6/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_6/zeros_1/packedq
lstm_6/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_6/zeros_1/Const?
lstm_6/zeros_1Filllstm_6/zeros_1/packed:output:0lstm_6/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_6/zeros_1?
lstm_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_6/transpose/perm?
lstm_6/transpose	Transposelstm_5/transpose_1:y:0lstm_6/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
lstm_6/transposed
lstm_6/Shape_1Shapelstm_6/transpose:y:0*
T0*
_output_shapes
:2
lstm_6/Shape_1?
lstm_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_6/strided_slice_1/stack?
lstm_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_1/stack_1?
lstm_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_1/stack_2?
lstm_6/strided_slice_1StridedSlicelstm_6/Shape_1:output:0%lstm_6/strided_slice_1/stack:output:0'lstm_6/strided_slice_1/stack_1:output:0'lstm_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_6/strided_slice_1?
"lstm_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_6/TensorArrayV2/element_shape?
lstm_6/TensorArrayV2TensorListReserve+lstm_6/TensorArrayV2/element_shape:output:0lstm_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_6/TensorArrayV2?
<lstm_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2>
<lstm_6/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_6/transpose:y:0Elstm_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_6/TensorArrayUnstack/TensorListFromTensor?
lstm_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_6/strided_slice_2/stack?
lstm_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_2/stack_1?
lstm_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_2/stack_2?
lstm_6/strided_slice_2StridedSlicelstm_6/transpose:y:0%lstm_6/strided_slice_2/stack:output:0'lstm_6/strided_slice_2/stack_1:output:0'lstm_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_6/strided_slice_2?
(lstm_6/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp1lstm_6_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02*
(lstm_6/lstm_cell_8/MatMul/ReadVariableOp?
lstm_6/lstm_cell_8/MatMulMatMullstm_6/strided_slice_2:output:00lstm_6/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_6/lstm_cell_8/MatMul?
*lstm_6/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp3lstm_6_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02,
*lstm_6/lstm_cell_8/MatMul_1/ReadVariableOp?
lstm_6/lstm_cell_8/MatMul_1MatMullstm_6/zeros:output:02lstm_6/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_6/lstm_cell_8/MatMul_1?
lstm_6/lstm_cell_8/addAddV2#lstm_6/lstm_cell_8/MatMul:product:0%lstm_6/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_6/lstm_cell_8/add?
)lstm_6/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp2lstm_6_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02+
)lstm_6/lstm_cell_8/BiasAdd/ReadVariableOp?
lstm_6/lstm_cell_8/BiasAddBiasAddlstm_6/lstm_cell_8/add:z:01lstm_6/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_6/lstm_cell_8/BiasAddv
lstm_6/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/lstm_cell_8/Const?
"lstm_6/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_6/lstm_cell_8/split/split_dim?
lstm_6/lstm_cell_8/splitSplit+lstm_6/lstm_cell_8/split/split_dim:output:0#lstm_6/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_6/lstm_cell_8/split?
lstm_6/lstm_cell_8/SigmoidSigmoid!lstm_6/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/Sigmoid?
lstm_6/lstm_cell_8/Sigmoid_1Sigmoid!lstm_6/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/Sigmoid_1?
lstm_6/lstm_cell_8/mulMul lstm_6/lstm_cell_8/Sigmoid_1:y:0lstm_6/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/mul?
lstm_6/lstm_cell_8/ReluRelu!lstm_6/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/Relu?
lstm_6/lstm_cell_8/mul_1Mullstm_6/lstm_cell_8/Sigmoid:y:0%lstm_6/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/mul_1?
lstm_6/lstm_cell_8/add_1AddV2lstm_6/lstm_cell_8/mul:z:0lstm_6/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/add_1?
lstm_6/lstm_cell_8/Sigmoid_2Sigmoid!lstm_6/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/Sigmoid_2?
lstm_6/lstm_cell_8/Relu_1Relulstm_6/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/Relu_1?
lstm_6/lstm_cell_8/mul_2Mul lstm_6/lstm_cell_8/Sigmoid_2:y:0'lstm_6/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/mul_2?
$lstm_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$lstm_6/TensorArrayV2_1/element_shape?
lstm_6/TensorArrayV2_1TensorListReserve-lstm_6/TensorArrayV2_1/element_shape:output:0lstm_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_6/TensorArrayV2_1\
lstm_6/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_6/time?
lstm_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_6/while/maximum_iterationsx
lstm_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_6/while/loop_counter?
lstm_6/whileWhile"lstm_6/while/loop_counter:output:0(lstm_6/while/maximum_iterations:output:0lstm_6/time:output:0lstm_6/TensorArrayV2_1:handle:0lstm_6/zeros:output:0lstm_6/zeros_1:output:0lstm_6/strided_slice_1:output:0>lstm_6/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_6_lstm_cell_8_matmul_readvariableop_resource3lstm_6_lstm_cell_8_matmul_1_readvariableop_resource2lstm_6_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_6_while_body_203541*$
condR
lstm_6_while_cond_203540*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
lstm_6/while?
7lstm_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7lstm_6/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_6/TensorArrayV2Stack/TensorListStackTensorListStacklstm_6/while:output:3@lstm_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02+
)lstm_6/TensorArrayV2Stack/TensorListStack?
lstm_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_6/strided_slice_3/stack?
lstm_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_6/strided_slice_3/stack_1?
lstm_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_3/stack_2?
lstm_6/strided_slice_3StridedSlice2lstm_6/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_6/strided_slice_3/stack:output:0'lstm_6/strided_slice_3/stack_1:output:0'lstm_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_6/strided_slice_3?
lstm_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_6/transpose_1/perm?
lstm_6/transpose_1	Transpose2lstm_6/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_6/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
lstm_6/transpose_1t
lstm_6/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_6/runtime?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMullstm_6/strided_slice_3:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/BiasAdd?
IdentityIdentitydense_12/BiasAdd:output:0^lstm_5/while^lstm_6/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::2
lstm_5/whilelstm_5/while2
lstm_6/whilelstm_6/while:` \
8
_output_shapes&
$:"??????????????????

 
_user_specified_nameinputs
?
O
3__inference_time_distributed_3_layer_call_fn_204325

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_2010122
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:"?????????????????? :` \
8
_output_shapes&
$:"?????????????????? 
 
_user_specified_nameinputs
?
?
D__inference_conv1d_2_layer_call_and_return_conditional_losses_200596

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????	@*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????	@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????	@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????	@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????	@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????
:::S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
while_cond_201546
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_201546___redundant_placeholder04
0while_while_cond_201546___redundant_placeholder14
0while_while_cond_201546___redundant_placeholder24
0while_while_cond_201546___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?W
?
B__inference_lstm_6_layer_call_and_return_conditional_losses_202955

inputs.
*lstm_cell_8_matmul_readvariableop_resource0
,lstm_cell_8_matmul_1_readvariableop_resource/
+lstm_cell_8_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOp?
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/MatMul?
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp?
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/MatMul_1?
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/add?
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/BiasAddh
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/Const|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dim?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_8/split?
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid?
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid_1?
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Relu?
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mul_1?
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/add_1?
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Relu_1?
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_202870*
condR
while_cond_202869*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::2
whilewhile:\ X
4
_output_shapes"
 :??????????????????2
 
_user_specified_nameinputs
?
?
D__inference_conv1d_3_layer_call_and_return_conditional_losses_205697

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????	@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????	@:::S O
+
_output_shapes
:?????????	@
 
_user_specified_nameinputs
?D
?
B__inference_lstm_6_layer_call_and_return_conditional_losses_202226

inputs
lstm_cell_8_202144
lstm_cell_8_202146
lstm_cell_8_202148
identity??#lstm_cell_8/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_202144lstm_cell_8_202146lstm_cell_8_202148*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_2017312%
#lstm_cell_8/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_202144lstm_cell_8_202146lstm_cell_8_202148*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_202157*
condR
while_cond_202156*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_8/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????2
 
_user_specified_nameinputs
?	
?
lstm_6_while_cond_203933*
&lstm_6_while_lstm_6_while_loop_counter0
,lstm_6_while_lstm_6_while_maximum_iterations
lstm_6_while_placeholder
lstm_6_while_placeholder_1
lstm_6_while_placeholder_2
lstm_6_while_placeholder_3,
(lstm_6_while_less_lstm_6_strided_slice_1B
>lstm_6_while_lstm_6_while_cond_203933___redundant_placeholder0B
>lstm_6_while_lstm_6_while_cond_203933___redundant_placeholder1B
>lstm_6_while_lstm_6_while_cond_203933___redundant_placeholder2B
>lstm_6_while_lstm_6_while_cond_203933___redundant_placeholder3
lstm_6_while_identity
?
lstm_6/while/LessLesslstm_6_while_placeholder(lstm_6_while_less_lstm_6_strided_slice_1*
T0*
_output_shapes
: 2
lstm_6/while/Lessr
lstm_6/while/IdentityIdentitylstm_6/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_6/while/Identity"7
lstm_6_while_identitylstm_6/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_201731

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????2:?????????:?????????::::O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
?
~
)__inference_conv1d_3_layer_call_fn_205706

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_2007272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????	@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????	@
 
_user_specified_nameinputs
?W
?
B__inference_lstm_6_layer_call_and_return_conditional_losses_205134

inputs.
*lstm_cell_8_matmul_readvariableop_resource0
,lstm_cell_8_matmul_1_readvariableop_resource/
+lstm_cell_8_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOp?
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/MatMul?
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp?
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/MatMul_1?
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/add?
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/BiasAddh
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/Const|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dim?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_8/split?
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid?
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid_1?
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Relu?
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mul_1?
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/add_1?
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Relu_1?
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_205049*
condR
while_cond_205048*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::2
whilewhile:\ X
4
_output_shapes"
 :??????????????????2
 
_user_specified_nameinputs
?
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_200901

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slices
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:????????? 2	
Reshape?
max_pooling1d_1/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_2008402!
max_pooling1d_1/PartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/3?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape(max_pooling1d_1/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2
	Reshape_1w
IdentityIdentityReshape_1:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:"?????????????????? :` \
8
_output_shapes&
$:"?????????????????? 
 
_user_specified_nameinputs
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_204111

inputs8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slices
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????
      2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????
2	
Reshape?
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDimsReshape:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????	@*
paddingVALID*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????	@*
squeeze_dims

?????????2
conv1d_2/conv1d/Squeeze?
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp?
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????	@2
conv1d_2/BiasAddw
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????	@2
conv1d_2/Reluq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape_1/shape/3?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshapeconv1d_2/Relu:activations:0Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????	@2
	Reshape_1w
IdentityIdentityReshape_1:output:0*
T0*8
_output_shapes&
$:"??????????????????	@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:"??????????????????
:::` \
8
_output_shapes&
$:"??????????????????

 
_user_specified_nameinputs
?
?
'__inference_lstm_5_layer_call_fn_204642
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_5_layer_call_and_return_conditional_losses_2014842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?D
?
B__inference_lstm_5_layer_call_and_return_conditional_losses_201484

inputs
lstm_cell_7_201402
lstm_cell_7_201404
lstm_cell_7_201406
identity??#lstm_cell_7/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_7/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_7_201402lstm_cell_7_201404lstm_cell_7_201406*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_2010882%
#lstm_cell_7/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_7_201402lstm_cell_7_201404lstm_cell_7_201406*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_201415*
condR
while_cond_201414*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_7/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2J
#lstm_cell_7/StatefulPartitionedCall#lstm_cell_7/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_206220
file_prefix$
 assignvariableop_dense_12_kernel$
 assignvariableop_1_dense_12_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate.
*assignvariableop_7_time_distributed_kernel,
(assignvariableop_8_time_distributed_bias0
,assignvariableop_9_time_distributed_1_kernel/
+assignvariableop_10_time_distributed_1_bias1
-assignvariableop_11_lstm_5_lstm_cell_7_kernel;
7assignvariableop_12_lstm_5_lstm_cell_7_recurrent_kernel/
+assignvariableop_13_lstm_5_lstm_cell_7_bias1
-assignvariableop_14_lstm_6_lstm_cell_8_kernel;
7assignvariableop_15_lstm_6_lstm_cell_8_recurrent_kernel/
+assignvariableop_16_lstm_6_lstm_cell_8_bias
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_1.
*assignvariableop_21_adam_dense_12_kernel_m,
(assignvariableop_22_adam_dense_12_bias_m6
2assignvariableop_23_adam_time_distributed_kernel_m4
0assignvariableop_24_adam_time_distributed_bias_m8
4assignvariableop_25_adam_time_distributed_1_kernel_m6
2assignvariableop_26_adam_time_distributed_1_bias_m8
4assignvariableop_27_adam_lstm_5_lstm_cell_7_kernel_mB
>assignvariableop_28_adam_lstm_5_lstm_cell_7_recurrent_kernel_m6
2assignvariableop_29_adam_lstm_5_lstm_cell_7_bias_m8
4assignvariableop_30_adam_lstm_6_lstm_cell_8_kernel_mB
>assignvariableop_31_adam_lstm_6_lstm_cell_8_recurrent_kernel_m6
2assignvariableop_32_adam_lstm_6_lstm_cell_8_bias_m.
*assignvariableop_33_adam_dense_12_kernel_v,
(assignvariableop_34_adam_dense_12_bias_v6
2assignvariableop_35_adam_time_distributed_kernel_v4
0assignvariableop_36_adam_time_distributed_bias_v8
4assignvariableop_37_adam_time_distributed_1_kernel_v6
2assignvariableop_38_adam_time_distributed_1_bias_v8
4assignvariableop_39_adam_lstm_5_lstm_cell_7_kernel_vB
>assignvariableop_40_adam_lstm_5_lstm_cell_7_recurrent_kernel_v6
2assignvariableop_41_adam_lstm_5_lstm_cell_7_bias_v8
4assignvariableop_42_adam_lstm_6_lstm_cell_8_kernel_vB
>assignvariableop_43_adam_lstm_6_lstm_cell_8_recurrent_kernel_v6
2assignvariableop_44_adam_lstm_6_lstm_cell_8_bias_v
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_12_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_12_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp*assignvariableop_7_time_distributed_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp(assignvariableop_8_time_distributed_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_time_distributed_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp+assignvariableop_10_time_distributed_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp-assignvariableop_11_lstm_5_lstm_cell_7_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp7assignvariableop_12_lstm_5_lstm_cell_7_recurrent_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp+assignvariableop_13_lstm_5_lstm_cell_7_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp-assignvariableop_14_lstm_6_lstm_cell_8_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp7assignvariableop_15_lstm_6_lstm_cell_8_recurrent_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp+assignvariableop_16_lstm_6_lstm_cell_8_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_12_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_12_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp2assignvariableop_23_adam_time_distributed_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp0assignvariableop_24_adam_time_distributed_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_time_distributed_1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp2assignvariableop_26_adam_time_distributed_1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_lstm_5_lstm_cell_7_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp>assignvariableop_28_adam_lstm_5_lstm_cell_7_recurrent_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_lstm_5_lstm_cell_7_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_lstm_6_lstm_cell_8_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp>assignvariableop_31_adam_lstm_6_lstm_cell_8_recurrent_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp2assignvariableop_32_adam_lstm_6_lstm_cell_8_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_12_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_12_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp2assignvariableop_35_adam_time_distributed_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp0assignvariableop_36_adam_time_distributed_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_time_distributed_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp2assignvariableop_38_adam_time_distributed_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp4assignvariableop_39_adam_lstm_5_lstm_cell_7_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp>assignvariableop_40_adam_lstm_5_lstm_cell_7_recurrent_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp2assignvariableop_41_adam_lstm_5_lstm_cell_7_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp4assignvariableop_42_adam_lstm_6_lstm_cell_8_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp>assignvariableop_43_adam_lstm_6_lstm_cell_8_recurrent_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp2assignvariableop_44_adam_lstm_6_lstm_cell_8_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45?
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_205750

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:??????????:?????????2:?????????2::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
?
?
1__inference_time_distributed_layer_call_fn_204157

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"??????????????????	@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_2006932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*8
_output_shapes&
$:"??????????????????	@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:"??????????????????
::22
StatefulPartitionedCallStatefulPartitionedCall:` \
8
_output_shapes&
$:"??????????????????

 
_user_specified_nameinputs
?
?
while_cond_204720
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_204720___redundant_placeholder04
0while_while_cond_204720___redundant_placeholder14
0while_while_cond_204720___redundant_placeholder24
0while_while_cond_204720___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?W
?
B__inference_lstm_6_layer_call_and_return_conditional_losses_202802

inputs.
*lstm_cell_8_matmul_readvariableop_resource0
,lstm_cell_8_matmul_1_readvariableop_resource/
+lstm_cell_8_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOp?
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/MatMul?
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp?
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/MatMul_1?
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/add?
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/BiasAddh
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/Const|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dim?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_8/split?
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid?
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid_1?
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Relu?
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mul_1?
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/add_1?
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Relu_1?
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_202717*
condR
while_cond_202716*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::2
whilewhile:\ X
4
_output_shapes"
 :??????????????????2
 
_user_specified_nameinputs
?W
?
B__inference_lstm_6_layer_call_and_return_conditional_losses_205287

inputs.
*lstm_cell_8_matmul_readvariableop_resource0
,lstm_cell_8_matmul_1_readvariableop_resource/
+lstm_cell_8_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOp?
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/MatMul?
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp?
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/MatMul_1?
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/add?
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_8/BiasAddh
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/Const|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dim?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_8/split?
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid?
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid_1?
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Relu?
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mul_1?
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/add_1?
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/Relu_1?
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_8/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_205202*
condR
while_cond_205201*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::2
whilewhile:\ X
4
_output_shapes"
 :??????????????????2
 
_user_specified_nameinputs
?
?
while_cond_204392
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_204392___redundant_placeholder04
0while_while_cond_204392___redundant_placeholder14
0while_while_cond_204392___redundant_placeholder24
0while_while_cond_204392___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
,__inference_lstm_cell_7_layer_call_fn_205800

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_2010882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:??????????:?????????2:?????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
?
?
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_205850

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????2:?????????:?????????::::O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?
?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_204213

inputs8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slices
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????	   @   2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????	@2	
Reshape?
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_3/conv1d/ExpandDims/dim?
conv1d_3/conv1d/ExpandDims
ExpandDimsReshape:output:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????	@2
conv1d_3/conv1d/ExpandDims?
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dim?
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d_3/conv1d/ExpandDims_1?
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d_3/conv1d?
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d_3/conv1d/Squeeze?
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_3/BiasAdd/ReadVariableOp?
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
conv1d_3/BiasAddw
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
conv1d_3/Reluq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/3?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshapeconv1d_3/Relu:activations:0Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2
	Reshape_1w
IdentityIdentityReshape_1:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:"??????????????????	@:::` \
8
_output_shapes&
$:"??????????????????	@
 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_201698

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????2:?????????:?????????::::O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
?
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_205712

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_205783

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:??????????:?????????2:?????????2::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
?
?
D__inference_conv1d_3_layer_call_and_return_conditional_losses_200727

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????	@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????	@:::S O
+
_output_shapes
:?????????	@
 
_user_specified_nameinputs
??
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_204025

inputsI
Etime_distributed_conv1d_2_conv1d_expanddims_1_readvariableop_resource=
9time_distributed_conv1d_2_biasadd_readvariableop_resourceK
Gtime_distributed_1_conv1d_3_conv1d_expanddims_1_readvariableop_resource?
;time_distributed_1_conv1d_3_biasadd_readvariableop_resource5
1lstm_5_lstm_cell_7_matmul_readvariableop_resource7
3lstm_5_lstm_cell_7_matmul_1_readvariableop_resource6
2lstm_5_lstm_cell_7_biasadd_readvariableop_resource5
1lstm_6_lstm_cell_8_matmul_readvariableop_resource7
3lstm_6_lstm_cell_8_matmul_1_readvariableop_resource6
2lstm_6_lstm_cell_8_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity??lstm_5/while?lstm_6/whilef
time_distributed/ShapeShapeinputs*
T0*
_output_shapes
:2
time_distributed/Shape?
$time_distributed/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$time_distributed/strided_slice/stack?
&time_distributed/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed/strided_slice/stack_1?
&time_distributed/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed/strided_slice/stack_2?
time_distributed/strided_sliceStridedSlicetime_distributed/Shape:output:0-time_distributed/strided_slice/stack:output:0/time_distributed/strided_slice/stack_1:output:0/time_distributed/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
time_distributed/strided_slice?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????
      2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????
2
time_distributed/Reshape?
/time_distributed/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/time_distributed/conv1d_2/conv1d/ExpandDims/dim?
+time_distributed/conv1d_2/conv1d/ExpandDims
ExpandDims!time_distributed/Reshape:output:08time_distributed/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
2-
+time_distributed/conv1d_2/conv1d/ExpandDims?
<time_distributed/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEtime_distributed_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<time_distributed/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
1time_distributed/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1time_distributed/conv1d_2/conv1d/ExpandDims_1/dim?
-time_distributed/conv1d_2/conv1d/ExpandDims_1
ExpandDimsDtime_distributed/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0:time_distributed/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-time_distributed/conv1d_2/conv1d/ExpandDims_1?
 time_distributed/conv1d_2/conv1dConv2D4time_distributed/conv1d_2/conv1d/ExpandDims:output:06time_distributed/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????	@*
paddingVALID*
strides
2"
 time_distributed/conv1d_2/conv1d?
(time_distributed/conv1d_2/conv1d/SqueezeSqueeze)time_distributed/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:?????????	@*
squeeze_dims

?????????2*
(time_distributed/conv1d_2/conv1d/Squeeze?
0time_distributed/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp9time_distributed_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0time_distributed/conv1d_2/BiasAdd/ReadVariableOp?
!time_distributed/conv1d_2/BiasAddBiasAdd1time_distributed/conv1d_2/conv1d/Squeeze:output:08time_distributed/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????	@2#
!time_distributed/conv1d_2/BiasAdd?
time_distributed/conv1d_2/ReluRelu*time_distributed/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????	@2 
time_distributed/conv1d_2/Relu?
"time_distributed/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"time_distributed/Reshape_1/shape/0?
"time_distributed/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2$
"time_distributed/Reshape_1/shape/2?
"time_distributed/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2$
"time_distributed/Reshape_1/shape/3?
 time_distributed/Reshape_1/shapePack+time_distributed/Reshape_1/shape/0:output:0'time_distributed/strided_slice:output:0+time_distributed/Reshape_1/shape/2:output:0+time_distributed/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 time_distributed/Reshape_1/shape?
time_distributed/Reshape_1Reshape,time_distributed/conv1d_2/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????	@2
time_distributed/Reshape_1?
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????
      2"
 time_distributed/Reshape_2/shape?
time_distributed/Reshape_2Reshapeinputs)time_distributed/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????
2
time_distributed/Reshape_2?
time_distributed_1/ShapeShape#time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:2
time_distributed_1/Shape?
&time_distributed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed_1/strided_slice/stack?
(time_distributed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_1/strided_slice/stack_1?
(time_distributed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_1/strided_slice/stack_2?
 time_distributed_1/strided_sliceStridedSlice!time_distributed_1/Shape:output:0/time_distributed_1/strided_slice/stack:output:01time_distributed_1/strided_slice/stack_1:output:01time_distributed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 time_distributed_1/strided_slice?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????	   @   2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????	@2
time_distributed_1/Reshape?
1time_distributed_1/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1time_distributed_1/conv1d_3/conv1d/ExpandDims/dim?
-time_distributed_1/conv1d_3/conv1d/ExpandDims
ExpandDims#time_distributed_1/Reshape:output:0:time_distributed_1/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????	@2/
-time_distributed_1/conv1d_3/conv1d/ExpandDims?
>time_distributed_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpGtime_distributed_1_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02@
>time_distributed_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
3time_distributed_1/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3time_distributed_1/conv1d_3/conv1d/ExpandDims_1/dim?
/time_distributed_1/conv1d_3/conv1d/ExpandDims_1
ExpandDimsFtime_distributed_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0<time_distributed_1/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 21
/time_distributed_1/conv1d_3/conv1d/ExpandDims_1?
"time_distributed_1/conv1d_3/conv1dConv2D6time_distributed_1/conv1d_3/conv1d/ExpandDims:output:08time_distributed_1/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2$
"time_distributed_1/conv1d_3/conv1d?
*time_distributed_1/conv1d_3/conv1d/SqueezeSqueeze+time_distributed_1/conv1d_3/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2,
*time_distributed_1/conv1d_3/conv1d/Squeeze?
2time_distributed_1/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp;time_distributed_1_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2time_distributed_1/conv1d_3/BiasAdd/ReadVariableOp?
#time_distributed_1/conv1d_3/BiasAddBiasAdd3time_distributed_1/conv1d_3/conv1d/Squeeze:output:0:time_distributed_1/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2%
#time_distributed_1/conv1d_3/BiasAdd?
 time_distributed_1/conv1d_3/ReluRelu,time_distributed_1/conv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2"
 time_distributed_1/conv1d_3/Relu?
$time_distributed_1/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$time_distributed_1/Reshape_1/shape/0?
$time_distributed_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$time_distributed_1/Reshape_1/shape/2?
$time_distributed_1/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2&
$time_distributed_1/Reshape_1/shape/3?
"time_distributed_1/Reshape_1/shapePack-time_distributed_1/Reshape_1/shape/0:output:0)time_distributed_1/strided_slice:output:0-time_distributed_1/Reshape_1/shape/2:output:0-time_distributed_1/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"time_distributed_1/Reshape_1/shape?
time_distributed_1/Reshape_1Reshape.time_distributed_1/conv1d_3/Relu:activations:0+time_distributed_1/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2
time_distributed_1/Reshape_1?
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????	   @   2$
"time_distributed_1/Reshape_2/shape?
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????	@2
time_distributed_1/Reshape_2?
time_distributed_2/ShapeShape%time_distributed_1/Reshape_1:output:0*
T0*
_output_shapes
:2
time_distributed_2/Shape?
&time_distributed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed_2/strided_slice/stack?
(time_distributed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_2/strided_slice/stack_1?
(time_distributed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_2/strided_slice/stack_2?
 time_distributed_2/strided_sliceStridedSlice!time_distributed_2/Shape:output:0/time_distributed_2/strided_slice/stack:output:01time_distributed_2/strided_slice/stack_1:output:01time_distributed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 time_distributed_2/strided_slice?
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2"
 time_distributed_2/Reshape/shape?
time_distributed_2/ReshapeReshape%time_distributed_1/Reshape_1:output:0)time_distributed_2/Reshape/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_2/Reshape?
1time_distributed_2/max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :23
1time_distributed_2/max_pooling1d_1/ExpandDims/dim?
-time_distributed_2/max_pooling1d_1/ExpandDims
ExpandDims#time_distributed_2/Reshape:output:0:time_distributed_2/max_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2/
-time_distributed_2/max_pooling1d_1/ExpandDims?
*time_distributed_2/max_pooling1d_1/MaxPoolMaxPool6time_distributed_2/max_pooling1d_1/ExpandDims:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2,
*time_distributed_2/max_pooling1d_1/MaxPool?
*time_distributed_2/max_pooling1d_1/SqueezeSqueeze3time_distributed_2/max_pooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims
2,
*time_distributed_2/max_pooling1d_1/Squeeze?
$time_distributed_2/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$time_distributed_2/Reshape_1/shape/0?
$time_distributed_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$time_distributed_2/Reshape_1/shape/2?
$time_distributed_2/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2&
$time_distributed_2/Reshape_1/shape/3?
"time_distributed_2/Reshape_1/shapePack-time_distributed_2/Reshape_1/shape/0:output:0)time_distributed_2/strided_slice:output:0-time_distributed_2/Reshape_1/shape/2:output:0-time_distributed_2/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"time_distributed_2/Reshape_1/shape?
time_distributed_2/Reshape_1Reshape3time_distributed_2/max_pooling1d_1/Squeeze:output:0+time_distributed_2/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2
time_distributed_2/Reshape_1?
"time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2$
"time_distributed_2/Reshape_2/shape?
time_distributed_2/Reshape_2Reshape%time_distributed_1/Reshape_1:output:0+time_distributed_2/Reshape_2/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_2/Reshape_2?
time_distributed_3/ShapeShape%time_distributed_2/Reshape_1:output:0*
T0*
_output_shapes
:2
time_distributed_3/Shape?
&time_distributed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&time_distributed_3/strided_slice/stack?
(time_distributed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_3/strided_slice/stack_1?
(time_distributed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(time_distributed_3/strided_slice/stack_2?
 time_distributed_3/strided_sliceStridedSlice!time_distributed_3/Shape:output:0/time_distributed_3/strided_slice/stack:output:01time_distributed_3/strided_slice/stack_1:output:01time_distributed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 time_distributed_3/strided_slice?
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2"
 time_distributed_3/Reshape/shape?
time_distributed_3/ReshapeReshape%time_distributed_2/Reshape_1:output:0)time_distributed_3/Reshape/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_3/Reshape?
"time_distributed_3/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2$
"time_distributed_3/flatten_1/Const?
$time_distributed_3/flatten_1/ReshapeReshape#time_distributed_3/Reshape:output:0+time_distributed_3/flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2&
$time_distributed_3/flatten_1/Reshape?
$time_distributed_3/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$time_distributed_3/Reshape_1/shape/0?
$time_distributed_3/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2&
$time_distributed_3/Reshape_1/shape/2?
"time_distributed_3/Reshape_1/shapePack-time_distributed_3/Reshape_1/shape/0:output:0)time_distributed_3/strided_slice:output:0-time_distributed_3/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2$
"time_distributed_3/Reshape_1/shape?
time_distributed_3/Reshape_1Reshape-time_distributed_3/flatten_1/Reshape:output:0+time_distributed_3/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:???????????????????2
time_distributed_3/Reshape_1?
"time_distributed_3/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2$
"time_distributed_3/Reshape_2/shape?
time_distributed_3/Reshape_2Reshape%time_distributed_2/Reshape_1:output:0+time_distributed_3/Reshape_2/shape:output:0*
T0*+
_output_shapes
:????????? 2
time_distributed_3/Reshape_2q
lstm_5/ShapeShape%time_distributed_3/Reshape_1:output:0*
T0*
_output_shapes
:2
lstm_5/Shape?
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice/stack?
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_1?
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_2?
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slicej
lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros/mul/y?
lstm_5/zeros/mulMullstm_5/strided_slice:output:0lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/mulm
lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros/Less/y?
lstm_5/zeros/LessLesslstm_5/zeros/mul:z:0lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/Lessp
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros/packed/1?
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros/packedm
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros/Const?
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/zerosn
lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros_1/mul/y?
lstm_5/zeros_1/mulMullstm_5/strided_slice:output:0lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/mulq
lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros_1/Less/y?
lstm_5/zeros_1/LessLesslstm_5/zeros_1/mul:z:0lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/Lesst
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros_1/packed/1?
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros_1/packedq
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros_1/Const?
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/zeros_1?
lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose/perm?
lstm_5/transpose	Transpose%time_distributed_3/Reshape_1:output:0lstm_5/transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
lstm_5/transposed
lstm_5/Shape_1Shapelstm_5/transpose:y:0*
T0*
_output_shapes
:2
lstm_5/Shape_1?
lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_1/stack?
lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_1?
lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_2?
lstm_5/strided_slice_1StridedSlicelstm_5/Shape_1:output:0%lstm_5/strided_slice_1/stack:output:0'lstm_5/strided_slice_1/stack_1:output:0'lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slice_1?
"lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_5/TensorArrayV2/element_shape?
lstm_5/TensorArrayV2TensorListReserve+lstm_5/TensorArrayV2/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2?
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2>
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_5/transpose:y:0Elstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_5/TensorArrayUnstack/TensorListFromTensor?
lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_2/stack?
lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_1?
lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_2?
lstm_5/strided_slice_2StridedSlicelstm_5/transpose:y:0%lstm_5/strided_slice_2/stack:output:0'lstm_5/strided_slice_2/stack_1:output:0'lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_5/strided_slice_2?
(lstm_5/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp1lstm_5_lstm_cell_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(lstm_5/lstm_cell_7/MatMul/ReadVariableOp?
lstm_5/lstm_cell_7/MatMulMatMullstm_5/strided_slice_2:output:00lstm_5/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_7/MatMul?
*lstm_5/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp3lstm_5_lstm_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02,
*lstm_5/lstm_cell_7/MatMul_1/ReadVariableOp?
lstm_5/lstm_cell_7/MatMul_1MatMullstm_5/zeros:output:02lstm_5/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_7/MatMul_1?
lstm_5/lstm_cell_7/addAddV2#lstm_5/lstm_cell_7/MatMul:product:0%lstm_5/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_7/add?
)lstm_5/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp2lstm_5_lstm_cell_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)lstm_5/lstm_cell_7/BiasAdd/ReadVariableOp?
lstm_5/lstm_cell_7/BiasAddBiasAddlstm_5/lstm_cell_7/add:z:01lstm_5/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_7/BiasAddv
lstm_5/lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/lstm_cell_7/Const?
"lstm_5/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_5/lstm_cell_7/split/split_dim?
lstm_5/lstm_cell_7/splitSplit+lstm_5/lstm_cell_7/split/split_dim:output:0#lstm_5/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_5/lstm_cell_7/split?
lstm_5/lstm_cell_7/SigmoidSigmoid!lstm_5/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/Sigmoid?
lstm_5/lstm_cell_7/Sigmoid_1Sigmoid!lstm_5/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/Sigmoid_1?
lstm_5/lstm_cell_7/mulMul lstm_5/lstm_cell_7/Sigmoid_1:y:0lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/mul?
lstm_5/lstm_cell_7/ReluRelu!lstm_5/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/Relu?
lstm_5/lstm_cell_7/mul_1Mullstm_5/lstm_cell_7/Sigmoid:y:0%lstm_5/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/mul_1?
lstm_5/lstm_cell_7/add_1AddV2lstm_5/lstm_cell_7/mul:z:0lstm_5/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/add_1?
lstm_5/lstm_cell_7/Sigmoid_2Sigmoid!lstm_5/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/Sigmoid_2?
lstm_5/lstm_cell_7/Relu_1Relulstm_5/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/Relu_1?
lstm_5/lstm_cell_7/mul_2Mul lstm_5/lstm_cell_7/Sigmoid_2:y:0'lstm_5/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_7/mul_2?
$lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2&
$lstm_5/TensorArrayV2_1/element_shape?
lstm_5/TensorArrayV2_1TensorListReserve-lstm_5/TensorArrayV2_1/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2_1\
lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/time?
lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_5/while/maximum_iterationsx
lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/while/loop_counter?
lstm_5/whileWhile"lstm_5/while/loop_counter:output:0(lstm_5/while/maximum_iterations:output:0lstm_5/time:output:0lstm_5/TensorArrayV2_1:handle:0lstm_5/zeros:output:0lstm_5/zeros_1:output:0lstm_5/strided_slice_1:output:0>lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_5_lstm_cell_7_matmul_readvariableop_resource3lstm_5_lstm_cell_7_matmul_1_readvariableop_resource2lstm_5_lstm_cell_7_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_5_while_body_203785*$
condR
lstm_5_while_cond_203784*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
lstm_5/while?
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_5/TensorArrayV2Stack/TensorListStackTensorListStacklstm_5/while:output:3@lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02+
)lstm_5/TensorArrayV2Stack/TensorListStack?
lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_5/strided_slice_3/stack?
lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_5/strided_slice_3/stack_1?
lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_3/stack_2?
lstm_5/strided_slice_3StridedSlice2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_5/strided_slice_3/stack:output:0'lstm_5/strided_slice_3/stack_1:output:0'lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_5/strided_slice_3?
lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose_1/perm?
lstm_5/transpose_1	Transpose2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_5/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
lstm_5/transpose_1t
lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/runtimeb
lstm_6/ShapeShapelstm_5/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_6/Shape?
lstm_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_6/strided_slice/stack?
lstm_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_6/strided_slice/stack_1?
lstm_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_6/strided_slice/stack_2?
lstm_6/strided_sliceStridedSlicelstm_6/Shape:output:0#lstm_6/strided_slice/stack:output:0%lstm_6/strided_slice/stack_1:output:0%lstm_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_6/strided_slicej
lstm_6/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/zeros/mul/y?
lstm_6/zeros/mulMullstm_6/strided_slice:output:0lstm_6/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros/mulm
lstm_6/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/zeros/Less/y?
lstm_6/zeros/LessLesslstm_6/zeros/mul:z:0lstm_6/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros/Lessp
lstm_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/zeros/packed/1?
lstm_6/zeros/packedPacklstm_6/strided_slice:output:0lstm_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_6/zeros/packedm
lstm_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_6/zeros/Const?
lstm_6/zerosFilllstm_6/zeros/packed:output:0lstm_6/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_6/zerosn
lstm_6/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/zeros_1/mul/y?
lstm_6/zeros_1/mulMullstm_6/strided_slice:output:0lstm_6/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros_1/mulq
lstm_6/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/zeros_1/Less/y?
lstm_6/zeros_1/LessLesslstm_6/zeros_1/mul:z:0lstm_6/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros_1/Lesst
lstm_6/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/zeros_1/packed/1?
lstm_6/zeros_1/packedPacklstm_6/strided_slice:output:0 lstm_6/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_6/zeros_1/packedq
lstm_6/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_6/zeros_1/Const?
lstm_6/zeros_1Filllstm_6/zeros_1/packed:output:0lstm_6/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_6/zeros_1?
lstm_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_6/transpose/perm?
lstm_6/transpose	Transposelstm_5/transpose_1:y:0lstm_6/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
lstm_6/transposed
lstm_6/Shape_1Shapelstm_6/transpose:y:0*
T0*
_output_shapes
:2
lstm_6/Shape_1?
lstm_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_6/strided_slice_1/stack?
lstm_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_1/stack_1?
lstm_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_1/stack_2?
lstm_6/strided_slice_1StridedSlicelstm_6/Shape_1:output:0%lstm_6/strided_slice_1/stack:output:0'lstm_6/strided_slice_1/stack_1:output:0'lstm_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_6/strided_slice_1?
"lstm_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_6/TensorArrayV2/element_shape?
lstm_6/TensorArrayV2TensorListReserve+lstm_6/TensorArrayV2/element_shape:output:0lstm_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_6/TensorArrayV2?
<lstm_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2>
<lstm_6/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_6/transpose:y:0Elstm_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_6/TensorArrayUnstack/TensorListFromTensor?
lstm_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_6/strided_slice_2/stack?
lstm_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_2/stack_1?
lstm_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_2/stack_2?
lstm_6/strided_slice_2StridedSlicelstm_6/transpose:y:0%lstm_6/strided_slice_2/stack:output:0'lstm_6/strided_slice_2/stack_1:output:0'lstm_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_6/strided_slice_2?
(lstm_6/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp1lstm_6_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02*
(lstm_6/lstm_cell_8/MatMul/ReadVariableOp?
lstm_6/lstm_cell_8/MatMulMatMullstm_6/strided_slice_2:output:00lstm_6/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_6/lstm_cell_8/MatMul?
*lstm_6/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp3lstm_6_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02,
*lstm_6/lstm_cell_8/MatMul_1/ReadVariableOp?
lstm_6/lstm_cell_8/MatMul_1MatMullstm_6/zeros:output:02lstm_6/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_6/lstm_cell_8/MatMul_1?
lstm_6/lstm_cell_8/addAddV2#lstm_6/lstm_cell_8/MatMul:product:0%lstm_6/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_6/lstm_cell_8/add?
)lstm_6/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp2lstm_6_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02+
)lstm_6/lstm_cell_8/BiasAdd/ReadVariableOp?
lstm_6/lstm_cell_8/BiasAddBiasAddlstm_6/lstm_cell_8/add:z:01lstm_6/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_6/lstm_cell_8/BiasAddv
lstm_6/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_6/lstm_cell_8/Const?
"lstm_6/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_6/lstm_cell_8/split/split_dim?
lstm_6/lstm_cell_8/splitSplit+lstm_6/lstm_cell_8/split/split_dim:output:0#lstm_6/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_6/lstm_cell_8/split?
lstm_6/lstm_cell_8/SigmoidSigmoid!lstm_6/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/Sigmoid?
lstm_6/lstm_cell_8/Sigmoid_1Sigmoid!lstm_6/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/Sigmoid_1?
lstm_6/lstm_cell_8/mulMul lstm_6/lstm_cell_8/Sigmoid_1:y:0lstm_6/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/mul?
lstm_6/lstm_cell_8/ReluRelu!lstm_6/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/Relu?
lstm_6/lstm_cell_8/mul_1Mullstm_6/lstm_cell_8/Sigmoid:y:0%lstm_6/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/mul_1?
lstm_6/lstm_cell_8/add_1AddV2lstm_6/lstm_cell_8/mul:z:0lstm_6/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/add_1?
lstm_6/lstm_cell_8/Sigmoid_2Sigmoid!lstm_6/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/Sigmoid_2?
lstm_6/lstm_cell_8/Relu_1Relulstm_6/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/Relu_1?
lstm_6/lstm_cell_8/mul_2Mul lstm_6/lstm_cell_8/Sigmoid_2:y:0'lstm_6/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_6/lstm_cell_8/mul_2?
$lstm_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$lstm_6/TensorArrayV2_1/element_shape?
lstm_6/TensorArrayV2_1TensorListReserve-lstm_6/TensorArrayV2_1/element_shape:output:0lstm_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_6/TensorArrayV2_1\
lstm_6/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_6/time?
lstm_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_6/while/maximum_iterationsx
lstm_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_6/while/loop_counter?
lstm_6/whileWhile"lstm_6/while/loop_counter:output:0(lstm_6/while/maximum_iterations:output:0lstm_6/time:output:0lstm_6/TensorArrayV2_1:handle:0lstm_6/zeros:output:0lstm_6/zeros_1:output:0lstm_6/strided_slice_1:output:0>lstm_6/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_6_lstm_cell_8_matmul_readvariableop_resource3lstm_6_lstm_cell_8_matmul_1_readvariableop_resource2lstm_6_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*$
bodyR
lstm_6_while_body_203934*$
condR
lstm_6_while_cond_203933*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
lstm_6/while?
7lstm_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7lstm_6/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_6/TensorArrayV2Stack/TensorListStackTensorListStacklstm_6/while:output:3@lstm_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02+
)lstm_6/TensorArrayV2Stack/TensorListStack?
lstm_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_6/strided_slice_3/stack?
lstm_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_6/strided_slice_3/stack_1?
lstm_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_6/strided_slice_3/stack_2?
lstm_6/strided_slice_3StridedSlice2lstm_6/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_6/strided_slice_3/stack:output:0'lstm_6/strided_slice_3/stack_1:output:0'lstm_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_6/strided_slice_3?
lstm_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_6/transpose_1/perm?
lstm_6/transpose_1	Transpose2lstm_6/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_6/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
lstm_6/transpose_1t
lstm_6/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_6/runtime?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMullstm_6/strided_slice_3:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/BiasAdd?
IdentityIdentitydense_12/BiasAdd:output:0^lstm_5/while^lstm_6/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::2
lstm_5/whilelstm_5/while2
lstm_6/whilelstm_6/while:` \
8
_output_shapes&
$:"??????????????????

 
_user_specified_nameinputs
?
?
'__inference_lstm_6_layer_call_fn_205309

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_6_layer_call_and_return_conditional_losses_2029552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????2
 
_user_specified_nameinputs
?D
?
B__inference_lstm_6_layer_call_and_return_conditional_losses_202094

inputs
lstm_cell_8_202012
lstm_cell_8_202014
lstm_cell_8_202016
identity??#lstm_cell_8/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_202012lstm_cell_8_202014lstm_cell_8_202016*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_2016982%
#lstm_cell_8/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_202012lstm_cell_8_202014lstm_cell_8_202016*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_202025*
condR
while_cond_202024*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_8/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????2
 
_user_specified_nameinputs
?$
?
while_body_202157
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_8_202181_0
while_lstm_cell_8_202183_0
while_lstm_cell_8_202185_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_8_202181
while_lstm_cell_8_202183
while_lstm_cell_8_202185??)while/lstm_cell_8/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_202181_0while_lstm_cell_8_202183_0while_lstm_cell_8_202185_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_2017312+
)while/lstm_cell_8/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_8/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1*^while/lstm_cell_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2*^while/lstm_cell_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_8_202181while_lstm_cell_8_202181_0"6
while_lstm_cell_8_202183while_lstm_cell_8_202183_0"6
while_lstm_cell_8_202185while_lstm_cell_8_202185_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2V
)while/lstm_cell_8/StatefulPartitionedCall)while/lstm_cell_8/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
~
)__inference_conv1d_2_layer_call_fn_205681

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????	@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_2005962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????	@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_205883

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????2:?????????:?????????::::O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?9
?
while_body_202717
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_8_matmul_readvariableop_resource_08
4while_lstm_cell_8_matmul_1_readvariableop_resource_07
3while_lstm_cell_8_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_8_matmul_readvariableop_resource6
2while_lstm_cell_8_matmul_1_readvariableop_resource5
1while_lstm_cell_8_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp?
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/MatMul?
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp?
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/MatMul_1?
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/add?
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp?
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/BiasAddt
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_8/Const?
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_8/split?
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid?
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid_1?
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul?
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Relu?
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul_1?
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/add_1?
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid_2?
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Relu_1?
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_202156
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_202156___redundant_placeholder04
0while_while_cond_202156___redundant_placeholder14
0while_while_cond_202156___redundant_placeholder24
0while_while_cond_202156___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_200824

inputs
conv1d_3_200813
conv1d_3_200815
identity?? conv1d_3/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slices
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????	   @   2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????	@2	
Reshape?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv1d_3_200813conv1d_3_200815*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_2007272"
 conv1d_3/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/3?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape)conv1d_3/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2
	Reshape_1?
IdentityIdentityReshape_1:output:0!^conv1d_3/StatefulPartitionedCall*
T0*8
_output_shapes&
$:"?????????????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:"??????????????????	@::2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall:` \
8
_output_shapes&
$:"??????????????????	@
 
_user_specified_nameinputs
?
?
while_cond_202024
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_202024___redundant_placeholder04
0while_while_cond_202024___redundant_placeholder14
0while_while_cond_202024___redundant_placeholder24
0while_while_cond_202024___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?B
?
lstm_5_while_body_203785*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3)
%lstm_5_while_lstm_5_strided_slice_1_0e
alstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_5_while_lstm_cell_7_matmul_readvariableop_resource_0?
;lstm_5_while_lstm_cell_7_matmul_1_readvariableop_resource_0>
:lstm_5_while_lstm_cell_7_biasadd_readvariableop_resource_0
lstm_5_while_identity
lstm_5_while_identity_1
lstm_5_while_identity_2
lstm_5_while_identity_3
lstm_5_while_identity_4
lstm_5_while_identity_5'
#lstm_5_while_lstm_5_strided_slice_1c
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor;
7lstm_5_while_lstm_cell_7_matmul_readvariableop_resource=
9lstm_5_while_lstm_cell_7_matmul_1_readvariableop_resource<
8lstm_5_while_lstm_cell_7_biasadd_readvariableop_resource??
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2@
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0lstm_5_while_placeholderGlstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype022
0lstm_5/while/TensorArrayV2Read/TensorListGetItem?
.lstm_5/while/lstm_cell_7/MatMul/ReadVariableOpReadVariableOp9lstm_5_while_lstm_cell_7_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype020
.lstm_5/while/lstm_cell_7/MatMul/ReadVariableOp?
lstm_5/while/lstm_cell_7/MatMulMatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_5/while/lstm_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
lstm_5/while/lstm_cell_7/MatMul?
0lstm_5/while/lstm_cell_7/MatMul_1/ReadVariableOpReadVariableOp;lstm_5_while_lstm_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype022
0lstm_5/while/lstm_cell_7/MatMul_1/ReadVariableOp?
!lstm_5/while/lstm_cell_7/MatMul_1MatMullstm_5_while_placeholder_28lstm_5/while/lstm_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_5/while/lstm_cell_7/MatMul_1?
lstm_5/while/lstm_cell_7/addAddV2)lstm_5/while/lstm_cell_7/MatMul:product:0+lstm_5/while/lstm_cell_7/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_5/while/lstm_cell_7/add?
/lstm_5/while/lstm_cell_7/BiasAdd/ReadVariableOpReadVariableOp:lstm_5_while_lstm_cell_7_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/lstm_5/while/lstm_cell_7/BiasAdd/ReadVariableOp?
 lstm_5/while/lstm_cell_7/BiasAddBiasAdd lstm_5/while/lstm_cell_7/add:z:07lstm_5/while/lstm_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 lstm_5/while/lstm_cell_7/BiasAdd?
lstm_5/while/lstm_cell_7/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_5/while/lstm_cell_7/Const?
(lstm_5/while/lstm_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_5/while/lstm_cell_7/split/split_dim?
lstm_5/while/lstm_cell_7/splitSplit1lstm_5/while/lstm_cell_7/split/split_dim:output:0)lstm_5/while/lstm_cell_7/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2 
lstm_5/while/lstm_cell_7/split?
 lstm_5/while/lstm_cell_7/SigmoidSigmoid'lstm_5/while/lstm_cell_7/split:output:0*
T0*'
_output_shapes
:?????????22"
 lstm_5/while/lstm_cell_7/Sigmoid?
"lstm_5/while/lstm_cell_7/Sigmoid_1Sigmoid'lstm_5/while/lstm_cell_7/split:output:1*
T0*'
_output_shapes
:?????????22$
"lstm_5/while/lstm_cell_7/Sigmoid_1?
lstm_5/while/lstm_cell_7/mulMul&lstm_5/while/lstm_cell_7/Sigmoid_1:y:0lstm_5_while_placeholder_3*
T0*'
_output_shapes
:?????????22
lstm_5/while/lstm_cell_7/mul?
lstm_5/while/lstm_cell_7/ReluRelu'lstm_5/while/lstm_cell_7/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_5/while/lstm_cell_7/Relu?
lstm_5/while/lstm_cell_7/mul_1Mul$lstm_5/while/lstm_cell_7/Sigmoid:y:0+lstm_5/while/lstm_cell_7/Relu:activations:0*
T0*'
_output_shapes
:?????????22 
lstm_5/while/lstm_cell_7/mul_1?
lstm_5/while/lstm_cell_7/add_1AddV2 lstm_5/while/lstm_cell_7/mul:z:0"lstm_5/while/lstm_cell_7/mul_1:z:0*
T0*'
_output_shapes
:?????????22 
lstm_5/while/lstm_cell_7/add_1?
"lstm_5/while/lstm_cell_7/Sigmoid_2Sigmoid'lstm_5/while/lstm_cell_7/split:output:3*
T0*'
_output_shapes
:?????????22$
"lstm_5/while/lstm_cell_7/Sigmoid_2?
lstm_5/while/lstm_cell_7/Relu_1Relu"lstm_5/while/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22!
lstm_5/while/lstm_cell_7/Relu_1?
lstm_5/while/lstm_cell_7/mul_2Mul&lstm_5/while/lstm_cell_7/Sigmoid_2:y:0-lstm_5/while/lstm_cell_7/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22 
lstm_5/while/lstm_cell_7/mul_2?
1lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_5_while_placeholder_1lstm_5_while_placeholder"lstm_5/while/lstm_cell_7/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_5/while/TensorArrayV2Write/TensorListSetItemj
lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add/y?
lstm_5/while/addAddV2lstm_5_while_placeholderlstm_5/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/addn
lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add_1/y?
lstm_5/while/add_1AddV2&lstm_5_while_lstm_5_while_loop_counterlstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/add_1s
lstm_5/while/IdentityIdentitylstm_5/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_5/while/Identity?
lstm_5/while/Identity_1Identity,lstm_5_while_lstm_5_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_5/while/Identity_1u
lstm_5/while/Identity_2Identitylstm_5/while/add:z:0*
T0*
_output_shapes
: 2
lstm_5/while/Identity_2?
lstm_5/while/Identity_3IdentityAlstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_5/while/Identity_3?
lstm_5/while/Identity_4Identity"lstm_5/while/lstm_cell_7/mul_2:z:0*
T0*'
_output_shapes
:?????????22
lstm_5/while/Identity_4?
lstm_5/while/Identity_5Identity"lstm_5/while/lstm_cell_7/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_5/while/Identity_5"7
lstm_5_while_identitylstm_5/while/Identity:output:0";
lstm_5_while_identity_1 lstm_5/while/Identity_1:output:0";
lstm_5_while_identity_2 lstm_5/while/Identity_2:output:0";
lstm_5_while_identity_3 lstm_5/while/Identity_3:output:0";
lstm_5_while_identity_4 lstm_5/while/Identity_4:output:0";
lstm_5_while_identity_5 lstm_5/while/Identity_5:output:0"L
#lstm_5_while_lstm_5_strided_slice_1%lstm_5_while_lstm_5_strided_slice_1_0"v
8lstm_5_while_lstm_cell_7_biasadd_readvariableop_resource:lstm_5_while_lstm_cell_7_biasadd_readvariableop_resource_0"x
9lstm_5_while_lstm_cell_7_matmul_1_readvariableop_resource;lstm_5_while_lstm_cell_7_matmul_1_readvariableop_resource_0"t
7lstm_5_while_lstm_cell_7_matmul_readvariableop_resource9lstm_5_while_lstm_cell_7_matmul_readvariableop_resource_0"?
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensoralstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_204251

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slices
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????       2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:????????? 2	
Reshape?
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_1/ExpandDims/dim?
max_pooling1d_1/ExpandDims
ExpandDimsReshape:output:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2
max_pooling1d_1/ExpandDims?
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling1d_1/MaxPool?
max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims
2
max_pooling1d_1/Squeezeq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/3?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape max_pooling1d_1/Squeeze:output:0Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2
	Reshape_1w
IdentityIdentityReshape_1:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:"?????????????????? :` \
8
_output_shapes&
$:"?????????????????? 
 
_user_specified_nameinputs
?9
?
while_body_202870
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_8_matmul_readvariableop_resource_08
4while_lstm_cell_8_matmul_1_readvariableop_resource_07
3while_lstm_cell_8_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_8_matmul_readvariableop_resource6
2while_lstm_cell_8_matmul_1_readvariableop_resource5
1while_lstm_cell_8_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOp?
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/MatMul?
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp?
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/MatMul_1?
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/add?
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOp?
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_8/BiasAddt
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_8/Const?
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_8/split?
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid?
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid_1?
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul?
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Relu?
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul_1?
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/add_1?
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Sigmoid_2?
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/Relu_1?
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_8/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_200663

inputs
conv1d_2_200652
conv1d_2_200654
identity?? conv1d_2/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slices
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????
      2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????
2	
Reshape?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv1d_2_200652conv1d_2_200654*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????	@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_2005962"
 conv1d_2/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape_1/shape/3?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape)conv1d_2/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????	@2
	Reshape_1?
IdentityIdentityReshape_1:output:0!^conv1d_2/StatefulPartitionedCall*
T0*8
_output_shapes&
$:"??????????????????	@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:"??????????????????
::2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall:` \
8
_output_shapes&
$:"??????????????????

 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
j
time_distributed_inputP
(serving_default_time_distributed_input:0"??????????????????
<
dense_120
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?S
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?P
_tf_keras_sequential?O{"class_name": "Sequential", "name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 10, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "time_distributed_input"}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 10, 1]}, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_3", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}}}, {"class_name": "LSTM", "config": {"name": "lstm_5", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTM", "config": {"name": "lstm_6", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 25, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 10, 1], "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 10, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 10, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "time_distributed_input"}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 10, 1]}, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_3", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}}}, {"class_name": "LSTM", "config": {"name": "lstm_5", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTM", "config": {"name": "lstm_6", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 25, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": ["mean_squared_error"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	layer
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_layer?
{"class_name": "TimeDistributed", "name": "time_distributed", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 10, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "time_distributed", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 10, 1]}, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 10, 1], "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 10, 1]}}
?

	layer
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "TimeDistributed", "name": "time_distributed_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "time_distributed_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 9, 64], "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 9, 64]}}
?
	layer
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TimeDistributed", "name": "time_distributed_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "time_distributed_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 8, 32], "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 8, 32]}}
?
	layer
trainable_variables
regularization_losses
 	variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TimeDistributed", "name": "time_distributed_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "time_distributed_3", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 4, 32], "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 4, 32]}}
?
"cell
#
state_spec
$trainable_variables
%regularization_losses
&	variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_5", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
?
(cell
)
state_spec
*trainable_variables
+regularization_losses
,	variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_6", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 25, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 50]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 50]}}
?

.kernel
/bias
0trainable_variables
1regularization_losses
2	variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 25}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25]}}
?
4iter

5beta_1

6beta_2
	7decay
8learning_rate.m?/m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?.v?/v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?"
	optimizer
v
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
.10
/11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
.10
/11"
trackable_list_wrapper
?
Clayer_regularization_losses
Dnon_trainable_variables
	trainable_variables

regularization_losses
	variables
Elayer_metrics
Fmetrics

Glayers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?	

9kernel
:bias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}}
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?
Llayer_regularization_losses
Mnon_trainable_variables
trainable_variables
regularization_losses
	variables
Nlayer_metrics
Ometrics

Players
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?	

;kernel
<bias
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}}
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
?
Ulayer_regularization_losses
Vnon_trainable_variables
trainable_variables
regularization_losses
	variables
Wlayer_metrics
Xmetrics

Ylayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling1D", "name": "max_pooling1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
^layer_regularization_losses
_non_trainable_variables
trainable_variables
regularization_losses
	variables
`layer_metrics
ametrics

blayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
glayer_regularization_losses
hnon_trainable_variables
trainable_variables
regularization_losses
 	variables
ilayer_metrics
jmetrics

klayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

=kernel
>recurrent_kernel
?bias
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_7", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
?
player_regularization_losses
qnon_trainable_variables
$trainable_variables
%regularization_losses
&	variables
rlayer_metrics

sstates
tmetrics

ulayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

@kernel
Arecurrent_kernel
Bbias
vtrainable_variables
wregularization_losses
x	variables
y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_8", "trainable": true, "dtype": "float32", "units": 25, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
?
zlayer_regularization_losses
{non_trainable_variables
*trainable_variables
+regularization_losses
,	variables
|layer_metrics

}states
~metrics

layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2dense_12/kernel
:2dense_12/bias
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
0trainable_variables
1regularization_losses
2	variables
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
-:+@2time_distributed/kernel
#:!@2time_distributed/bias
/:-@ 2time_distributed_1/kernel
%:# 2time_distributed_1/bias
-:+
??2lstm_5/lstm_cell_7/kernel
6:4	2?2#lstm_5/lstm_cell_7/recurrent_kernel
&:$?2lstm_5/lstm_cell_7/bias
+:)2d2lstm_6/lstm_cell_8/kernel
5:3d2#lstm_6/lstm_cell_8/recurrent_kernel
%:#d2lstm_6/lstm_cell_8/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
Htrainable_variables
Iregularization_losses
J	variables
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
Qtrainable_variables
Rregularization_losses
S	variables
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
Ztrainable_variables
[regularization_losses
\	variables
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
ctrainable_variables
dregularization_losses
e	variables
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
ltrainable_variables
mregularization_losses
n	variables
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
"0"
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
vtrainable_variables
wregularization_losses
x	variables
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mean_squared_error", "dtype": "float32", "config": {"name": "mean_squared_error", "dtype": "float32", "fn": "mean_squared_error"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
&:$2Adam/dense_12/kernel/m
 :2Adam/dense_12/bias/m
2:0@2Adam/time_distributed/kernel/m
(:&@2Adam/time_distributed/bias/m
4:2@ 2 Adam/time_distributed_1/kernel/m
*:( 2Adam/time_distributed_1/bias/m
2:0
??2 Adam/lstm_5/lstm_cell_7/kernel/m
;:9	2?2*Adam/lstm_5/lstm_cell_7/recurrent_kernel/m
+:)?2Adam/lstm_5/lstm_cell_7/bias/m
0:.2d2 Adam/lstm_6/lstm_cell_8/kernel/m
::8d2*Adam/lstm_6/lstm_cell_8/recurrent_kernel/m
*:(d2Adam/lstm_6/lstm_cell_8/bias/m
&:$2Adam/dense_12/kernel/v
 :2Adam/dense_12/bias/v
2:0@2Adam/time_distributed/kernel/v
(:&@2Adam/time_distributed/bias/v
4:2@ 2 Adam/time_distributed_1/kernel/v
*:( 2Adam/time_distributed_1/bias/v
2:0
??2 Adam/lstm_5/lstm_cell_7/kernel/v
;:9	2?2*Adam/lstm_5/lstm_cell_7/recurrent_kernel/v
+:)?2Adam/lstm_5/lstm_cell_7/bias/v
0:.2d2 Adam/lstm_6/lstm_cell_8/kernel/v
::8d2*Adam/lstm_6/lstm_cell_8/recurrent_kernel/v
*:(d2Adam/lstm_6/lstm_cell_8/bias/v
?2?
!__inference__wrapped_model_200569?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *F?C
A?>
time_distributed_input"??????????????????

?2?
H__inference_sequential_5_layer_call_and_return_conditional_losses_203012
H__inference_sequential_5_layer_call_and_return_conditional_losses_204025
H__inference_sequential_5_layer_call_and_return_conditional_losses_203055
H__inference_sequential_5_layer_call_and_return_conditional_losses_203632?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_5_layer_call_fn_204083
-__inference_sequential_5_layer_call_fn_204054
-__inference_sequential_5_layer_call_fn_203200
-__inference_sequential_5_layer_call_fn_203128?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_time_distributed_layer_call_and_return_conditional_losses_204139
L__inference_time_distributed_layer_call_and_return_conditional_losses_204111?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
1__inference_time_distributed_layer_call_fn_204157
1__inference_time_distributed_layer_call_fn_204148?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_204185
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_204213?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_time_distributed_1_layer_call_fn_204231
3__inference_time_distributed_1_layer_call_fn_204222?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_204251
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_204271?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_time_distributed_2_layer_call_fn_204281
3__inference_time_distributed_2_layer_call_fn_204276?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_204298
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_204315?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_time_distributed_3_layer_call_fn_204320
3__inference_time_distributed_3_layer_call_fn_204325?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_lstm_5_layer_call_and_return_conditional_losses_204478
B__inference_lstm_5_layer_call_and_return_conditional_losses_204631
B__inference_lstm_5_layer_call_and_return_conditional_losses_204806
B__inference_lstm_5_layer_call_and_return_conditional_losses_204959?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_lstm_5_layer_call_fn_204653
'__inference_lstm_5_layer_call_fn_204970
'__inference_lstm_5_layer_call_fn_204981
'__inference_lstm_5_layer_call_fn_204642?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_lstm_6_layer_call_and_return_conditional_losses_205462
B__inference_lstm_6_layer_call_and_return_conditional_losses_205287
B__inference_lstm_6_layer_call_and_return_conditional_losses_205615
B__inference_lstm_6_layer_call_and_return_conditional_losses_205134?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_lstm_6_layer_call_fn_205298
'__inference_lstm_6_layer_call_fn_205626
'__inference_lstm_6_layer_call_fn_205637
'__inference_lstm_6_layer_call_fn_205309?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dense_12_layer_call_and_return_conditional_losses_205647?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_12_layer_call_fn_205656?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
BB@
$__inference_signature_wrapper_203239time_distributed_input
?2?
D__inference_conv1d_2_layer_call_and_return_conditional_losses_205672?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv1d_2_layer_call_fn_205681?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv1d_3_layer_call_and_return_conditional_losses_205697?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv1d_3_layer_call_fn_205706?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_200840?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
0__inference_max_pooling1d_1_layer_call_fn_200846?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
E__inference_flatten_1_layer_call_and_return_conditional_losses_205712?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_1_layer_call_fn_205717?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_205750
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_205783?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_lstm_cell_7_layer_call_fn_205800
,__inference_lstm_cell_7_layer_call_fn_205817?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_205850
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_205883?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_lstm_cell_8_layer_call_fn_205900
,__inference_lstm_cell_8_layer_call_fn_205917?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
!__inference__wrapped_model_200569?9:;<=>?@AB./P?M
F?C
A?>
time_distributed_input"??????????????????

? "3?0
.
dense_12"?
dense_12??????????
D__inference_conv1d_2_layer_call_and_return_conditional_losses_205672d9:3?0
)?&
$?!
inputs?????????

? ")?&
?
0?????????	@
? ?
)__inference_conv1d_2_layer_call_fn_205681W9:3?0
)?&
$?!
inputs?????????

? "??????????	@?
D__inference_conv1d_3_layer_call_and_return_conditional_losses_205697d;<3?0
)?&
$?!
inputs?????????	@
? ")?&
?
0????????? 
? ?
)__inference_conv1d_3_layer_call_fn_205706W;<3?0
)?&
$?!
inputs?????????	@
? "?????????? ?
D__inference_dense_12_layer_call_and_return_conditional_losses_205647\.//?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_12_layer_call_fn_205656O.//?,
%?"
 ?
inputs?????????
? "???????????
E__inference_flatten_1_layer_call_and_return_conditional_losses_205712]3?0
)?&
$?!
inputs????????? 
? "&?#
?
0??????????
? ~
*__inference_flatten_1_layer_call_fn_205717P3?0
)?&
$?!
inputs????????? 
? "????????????
B__inference_lstm_5_layer_call_and_return_conditional_losses_204478?=>?P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "2?/
(?%
0??????????????????2
? ?
B__inference_lstm_5_layer_call_and_return_conditional_losses_204631?=>?P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "2?/
(?%
0??????????????????2
? ?
B__inference_lstm_5_layer_call_and_return_conditional_losses_204806?=>?I?F
??<
.?+
inputs???????????????????

 
p

 
? "2?/
(?%
0??????????????????2
? ?
B__inference_lstm_5_layer_call_and_return_conditional_losses_204959?=>?I?F
??<
.?+
inputs???????????????????

 
p 

 
? "2?/
(?%
0??????????????????2
? ?
'__inference_lstm_5_layer_call_fn_204642~=>?P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "%?"??????????????????2?
'__inference_lstm_5_layer_call_fn_204653~=>?P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "%?"??????????????????2?
'__inference_lstm_5_layer_call_fn_204970w=>?I?F
??<
.?+
inputs???????????????????

 
p

 
? "%?"??????????????????2?
'__inference_lstm_5_layer_call_fn_204981w=>?I?F
??<
.?+
inputs???????????????????

 
p 

 
? "%?"??????????????????2?
B__inference_lstm_6_layer_call_and_return_conditional_losses_205134v@ABH?E
>?;
-?*
inputs??????????????????2

 
p

 
? "%?"
?
0?????????
? ?
B__inference_lstm_6_layer_call_and_return_conditional_losses_205287v@ABH?E
>?;
-?*
inputs??????????????????2

 
p 

 
? "%?"
?
0?????????
? ?
B__inference_lstm_6_layer_call_and_return_conditional_losses_205462}@ABO?L
E?B
4?1
/?,
inputs/0??????????????????2

 
p

 
? "%?"
?
0?????????
? ?
B__inference_lstm_6_layer_call_and_return_conditional_losses_205615}@ABO?L
E?B
4?1
/?,
inputs/0??????????????????2

 
p 

 
? "%?"
?
0?????????
? ?
'__inference_lstm_6_layer_call_fn_205298i@ABH?E
>?;
-?*
inputs??????????????????2

 
p

 
? "???????????
'__inference_lstm_6_layer_call_fn_205309i@ABH?E
>?;
-?*
inputs??????????????????2

 
p 

 
? "???????????
'__inference_lstm_6_layer_call_fn_205626p@ABO?L
E?B
4?1
/?,
inputs/0??????????????????2

 
p

 
? "???????????
'__inference_lstm_6_layer_call_fn_205637p@ABO?L
E?B
4?1
/?,
inputs/0??????????????????2

 
p 

 
? "???????????
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_205750?=>???~
w?t
!?
inputs??????????
K?H
"?
states/0?????????2
"?
states/1?????????2
p
? "s?p
i?f
?
0/0?????????2
E?B
?
0/1/0?????????2
?
0/1/1?????????2
? ?
G__inference_lstm_cell_7_layer_call_and_return_conditional_losses_205783?=>???~
w?t
!?
inputs??????????
K?H
"?
states/0?????????2
"?
states/1?????????2
p 
? "s?p
i?f
?
0/0?????????2
E?B
?
0/1/0?????????2
?
0/1/1?????????2
? ?
,__inference_lstm_cell_7_layer_call_fn_205800?=>???~
w?t
!?
inputs??????????
K?H
"?
states/0?????????2
"?
states/1?????????2
p
? "c?`
?
0?????????2
A?>
?
1/0?????????2
?
1/1?????????2?
,__inference_lstm_cell_7_layer_call_fn_205817?=>???~
w?t
!?
inputs??????????
K?H
"?
states/0?????????2
"?
states/1?????????2
p 
? "c?`
?
0?????????2
A?>
?
1/0?????????2
?
1/1?????????2?
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_205850?@AB??}
v?s
 ?
inputs?????????2
K?H
"?
states/0?????????
"?
states/1?????????
p
? "s?p
i?f
?
0/0?????????
E?B
?
0/1/0?????????
?
0/1/1?????????
? ?
G__inference_lstm_cell_8_layer_call_and_return_conditional_losses_205883?@AB??}
v?s
 ?
inputs?????????2
K?H
"?
states/0?????????
"?
states/1?????????
p 
? "s?p
i?f
?
0/0?????????
E?B
?
0/1/0?????????
?
0/1/1?????????
? ?
,__inference_lstm_cell_8_layer_call_fn_205900?@AB??}
v?s
 ?
inputs?????????2
K?H
"?
states/0?????????
"?
states/1?????????
p
? "c?`
?
0?????????
A?>
?
1/0?????????
?
1/1??????????
,__inference_lstm_cell_8_layer_call_fn_205917?@AB??}
v?s
 ?
inputs?????????2
K?H
"?
states/0?????????
"?
states/1?????????
p 
? "c?`
?
0?????????
A?>
?
1/0?????????
?
1/1??????????
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_200840?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
0__inference_max_pooling1d_1_layer_call_fn_200846wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
H__inference_sequential_5_layer_call_and_return_conditional_losses_203012?9:;<=>?@AB./X?U
N?K
A?>
time_distributed_input"??????????????????

p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_5_layer_call_and_return_conditional_losses_203055?9:;<=>?@AB./X?U
N?K
A?>
time_distributed_input"??????????????????

p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_5_layer_call_and_return_conditional_losses_2036329:;<=>?@AB./H?E
>?;
1?.
inputs"??????????????????

p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_5_layer_call_and_return_conditional_losses_2040259:;<=>?@AB./H?E
>?;
1?.
inputs"??????????????????

p 

 
? "%?"
?
0?????????
? ?
-__inference_sequential_5_layer_call_fn_203128?9:;<=>?@AB./X?U
N?K
A?>
time_distributed_input"??????????????????

p

 
? "???????????
-__inference_sequential_5_layer_call_fn_203200?9:;<=>?@AB./X?U
N?K
A?>
time_distributed_input"??????????????????

p 

 
? "???????????
-__inference_sequential_5_layer_call_fn_204054r9:;<=>?@AB./H?E
>?;
1?.
inputs"??????????????????

p

 
? "???????????
-__inference_sequential_5_layer_call_fn_204083r9:;<=>?@AB./H?E
>?;
1?.
inputs"??????????????????

p 

 
? "???????????
$__inference_signature_wrapper_203239?9:;<=>?@AB./j?g
? 
`?]
[
time_distributed_inputA?>
time_distributed_input"??????????????????
"3?0
.
dense_12"?
dense_12??????????
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_204185?;<H?E
>?;
1?.
inputs"??????????????????	@
p

 
? "6?3
,?)
0"?????????????????? 
? ?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_204213?;<H?E
>?;
1?.
inputs"??????????????????	@
p 

 
? "6?3
,?)
0"?????????????????? 
? ?
3__inference_time_distributed_1_layer_call_fn_204222y;<H?E
>?;
1?.
inputs"??????????????????	@
p

 
? ")?&"?????????????????? ?
3__inference_time_distributed_1_layer_call_fn_204231y;<H?E
>?;
1?.
inputs"??????????????????	@
p 

 
? ")?&"?????????????????? ?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_204251?H?E
>?;
1?.
inputs"?????????????????? 
p

 
? "6?3
,?)
0"?????????????????? 
? ?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_204271?H?E
>?;
1?.
inputs"?????????????????? 
p 

 
? "6?3
,?)
0"?????????????????? 
? ?
3__inference_time_distributed_2_layer_call_fn_204276uH?E
>?;
1?.
inputs"?????????????????? 
p

 
? ")?&"?????????????????? ?
3__inference_time_distributed_2_layer_call_fn_204281uH?E
>?;
1?.
inputs"?????????????????? 
p 

 
? ")?&"?????????????????? ?
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_204298H?E
>?;
1?.
inputs"?????????????????? 
p

 
? "3?0
)?&
0???????????????????
? ?
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_204315H?E
>?;
1?.
inputs"?????????????????? 
p 

 
? "3?0
)?&
0???????????????????
? ?
3__inference_time_distributed_3_layer_call_fn_204320rH?E
>?;
1?.
inputs"?????????????????? 
p

 
? "&?#????????????????????
3__inference_time_distributed_3_layer_call_fn_204325rH?E
>?;
1?.
inputs"?????????????????? 
p 

 
? "&?#????????????????????
L__inference_time_distributed_layer_call_and_return_conditional_losses_204111?9:H?E
>?;
1?.
inputs"??????????????????

p

 
? "6?3
,?)
0"??????????????????	@
? ?
L__inference_time_distributed_layer_call_and_return_conditional_losses_204139?9:H?E
>?;
1?.
inputs"??????????????????

p 

 
? "6?3
,?)
0"??????????????????	@
? ?
1__inference_time_distributed_layer_call_fn_204148y9:H?E
>?;
1?.
inputs"??????????????????

p

 
? ")?&"??????????????????	@?
1__inference_time_distributed_layer_call_fn_204157y9:H?E
>?;
1?.
inputs"??????????????????

p 

 
? ")?&"??????????????????	@