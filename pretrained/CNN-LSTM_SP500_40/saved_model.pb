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
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:(*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:(*
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
lstm_8/lstm_cell_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*+
shared_namelstm_8/lstm_cell_10/kernel
?
.lstm_8/lstm_cell_10/kernel/Read/ReadVariableOpReadVariableOplstm_8/lstm_cell_10/kernel* 
_output_shapes
:
??*
dtype0
?
$lstm_8/lstm_cell_10/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*5
shared_name&$lstm_8/lstm_cell_10/recurrent_kernel
?
8lstm_8/lstm_cell_10/recurrent_kernel/Read/ReadVariableOpReadVariableOp$lstm_8/lstm_cell_10/recurrent_kernel*
_output_shapes
:	2?*
dtype0
?
lstm_8/lstm_cell_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namelstm_8/lstm_cell_10/bias
?
,lstm_8/lstm_cell_10/bias/Read/ReadVariableOpReadVariableOplstm_8/lstm_cell_10/bias*
_output_shapes	
:?*
dtype0
?
lstm_9/lstm_cell_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2d*+
shared_namelstm_9/lstm_cell_11/kernel
?
.lstm_9/lstm_cell_11/kernel/Read/ReadVariableOpReadVariableOplstm_9/lstm_cell_11/kernel*
_output_shapes

:2d*
dtype0
?
$lstm_9/lstm_cell_11/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*5
shared_name&$lstm_9/lstm_cell_11/recurrent_kernel
?
8lstm_9/lstm_cell_11/recurrent_kernel/Read/ReadVariableOpReadVariableOp$lstm_9/lstm_cell_11/recurrent_kernel*
_output_shapes

:d*
dtype0
?
lstm_9/lstm_cell_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*)
shared_namelstm_9/lstm_cell_11/bias
?
,lstm_9/lstm_cell_11/bias/Read/ReadVariableOpReadVariableOplstm_9/lstm_cell_11/bias*
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
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes

:(*
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:(*
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
!Adam/lstm_8/lstm_cell_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*2
shared_name#!Adam/lstm_8/lstm_cell_10/kernel/m
?
5Adam/lstm_8/lstm_cell_10/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/lstm_8/lstm_cell_10/kernel/m* 
_output_shapes
:
??*
dtype0
?
+Adam/lstm_8/lstm_cell_10/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*<
shared_name-+Adam/lstm_8/lstm_cell_10/recurrent_kernel/m
?
?Adam/lstm_8/lstm_cell_10/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/lstm_8/lstm_cell_10/recurrent_kernel/m*
_output_shapes
:	2?*
dtype0
?
Adam/lstm_8/lstm_cell_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!Adam/lstm_8/lstm_cell_10/bias/m
?
3Adam/lstm_8/lstm_cell_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_8/lstm_cell_10/bias/m*
_output_shapes	
:?*
dtype0
?
!Adam/lstm_9/lstm_cell_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2d*2
shared_name#!Adam/lstm_9/lstm_cell_11/kernel/m
?
5Adam/lstm_9/lstm_cell_11/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/lstm_9/lstm_cell_11/kernel/m*
_output_shapes

:2d*
dtype0
?
+Adam/lstm_9/lstm_cell_11/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*<
shared_name-+Adam/lstm_9/lstm_cell_11/recurrent_kernel/m
?
?Adam/lstm_9/lstm_cell_11/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/lstm_9/lstm_cell_11/recurrent_kernel/m*
_output_shapes

:d*
dtype0
?
Adam/lstm_9/lstm_cell_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*0
shared_name!Adam/lstm_9/lstm_cell_11/bias/m
?
3Adam/lstm_9/lstm_cell_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_9/lstm_cell_11/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes

:(*
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:(*
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
!Adam/lstm_8/lstm_cell_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*2
shared_name#!Adam/lstm_8/lstm_cell_10/kernel/v
?
5Adam/lstm_8/lstm_cell_10/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/lstm_8/lstm_cell_10/kernel/v* 
_output_shapes
:
??*
dtype0
?
+Adam/lstm_8/lstm_cell_10/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*<
shared_name-+Adam/lstm_8/lstm_cell_10/recurrent_kernel/v
?
?Adam/lstm_8/lstm_cell_10/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/lstm_8/lstm_cell_10/recurrent_kernel/v*
_output_shapes
:	2?*
dtype0
?
Adam/lstm_8/lstm_cell_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!Adam/lstm_8/lstm_cell_10/bias/v
?
3Adam/lstm_8/lstm_cell_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_8/lstm_cell_10/bias/v*
_output_shapes	
:?*
dtype0
?
!Adam/lstm_9/lstm_cell_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2d*2
shared_name#!Adam/lstm_9/lstm_cell_11/kernel/v
?
5Adam/lstm_9/lstm_cell_11/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/lstm_9/lstm_cell_11/kernel/v*
_output_shapes

:2d*
dtype0
?
+Adam/lstm_9/lstm_cell_11/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*<
shared_name-+Adam/lstm_9/lstm_cell_11/recurrent_kernel/v
?
?Adam/lstm_9/lstm_cell_11/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/lstm_9/lstm_cell_11/recurrent_kernel/v*
_output_shapes

:d*
dtype0
?
Adam/lstm_9/lstm_cell_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*0
shared_name!Adam/lstm_9/lstm_cell_11/bias/v
?
3Adam/lstm_9/lstm_cell_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_9/lstm_cell_11/bias/v*
_output_shapes
:d*
dtype0

NoOpNoOp
?Q
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?P
value?PB?P B?P
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
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
]
	layer
regularization_losses
	variables
trainable_variables
	keras_api
]
	layer
regularization_losses
	variables
trainable_variables
	keras_api
]
	layer
regularization_losses
	variables
trainable_variables
	keras_api
]
	layer
regularization_losses
	variables
 trainable_variables
!	keras_api
l
"cell
#
state_spec
$regularization_losses
%	variables
&trainable_variables
'	keras_api
l
(cell
)
state_spec
*regularization_losses
+	variables
,trainable_variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
?
4iter

5beta_1

6beta_2
	7decay
8learning_rate.m?/m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?.v?/v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?
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
	regularization_losses

	variables

Clayers
Dmetrics
Elayer_regularization_losses
trainable_variables
Flayer_metrics
Gnon_trainable_variables
 
h

9kernel
:bias
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
 

90
:1

90
:1
?
regularization_losses
	variables

Llayers
Mlayer_regularization_losses
Nlayer_metrics
trainable_variables
Ometrics
Pnon_trainable_variables
h

;kernel
<bias
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
 

;0
<1

;0
<1
?
regularization_losses
	variables

Ulayers
Vlayer_regularization_losses
Wlayer_metrics
trainable_variables
Xmetrics
Ynon_trainable_variables
R
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
 
 
 
?
regularization_losses
	variables

^layers
_layer_regularization_losses
`layer_metrics
trainable_variables
ametrics
bnon_trainable_variables
R
cregularization_losses
d	variables
etrainable_variables
f	keras_api
 
 
 
?
regularization_losses
	variables

glayers
hlayer_regularization_losses
ilayer_metrics
 trainable_variables
jmetrics
knon_trainable_variables
~

=kernel
>recurrent_kernel
?bias
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
 
 

=0
>1
?2

=0
>1
?2
?
$regularization_losses
%	variables

players
qmetrics
rlayer_regularization_losses
&trainable_variables
slayer_metrics

tstates
unon_trainable_variables
~

@kernel
Arecurrent_kernel
Bbias
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
 
 

@0
A1
B2

@0
A1
B2
?
*regularization_losses
+	variables

zlayers
{metrics
|layer_regularization_losses
,trainable_variables
}layer_metrics

~states
non_trainable_variables
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
?
0regularization_losses
1	variables
?layers
 ?layer_regularization_losses
?layer_metrics
2trainable_variables
?metrics
?non_trainable_variables
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
SQ
VARIABLE_VALUEtime_distributed/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEtime_distributed/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtime_distributed_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtime_distributed_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElstm_8/lstm_cell_10/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$lstm_8/lstm_cell_10/recurrent_kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUElstm_8/lstm_cell_10/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElstm_9/lstm_cell_11/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$lstm_9/lstm_cell_11/recurrent_kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUElstm_9/lstm_cell_11/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
1
0
1
2
3
4
5
6

?0
?1
 
 
 
 

90
:1

90
:1
?
Hregularization_losses
I	variables
?layers
 ?layer_regularization_losses
?layer_metrics
Jtrainable_variables
?metrics
?non_trainable_variables

0
 
 
 
 
 

;0
<1

;0
<1
?
Qregularization_losses
R	variables
?layers
 ?layer_regularization_losses
?layer_metrics
Strainable_variables
?metrics
?non_trainable_variables

0
 
 
 
 
 
 
 
?
Zregularization_losses
[	variables
?layers
 ?layer_regularization_losses
?layer_metrics
\trainable_variables
?metrics
?non_trainable_variables

0
 
 
 
 
 
 
 
?
cregularization_losses
d	variables
?layers
 ?layer_regularization_losses
?layer_metrics
etrainable_variables
?metrics
?non_trainable_variables

0
 
 
 
 
 

=0
>1
?2

=0
>1
?2
?
lregularization_losses
m	variables
?layers
 ?layer_regularization_losses
?layer_metrics
ntrainable_variables
?metrics
?non_trainable_variables

"0
 
 
 
 
 
 

@0
A1
B2

@0
A1
B2
?
vregularization_losses
w	variables
?layers
 ?layer_regularization_losses
?layer_metrics
xtrainable_variables
?metrics
?non_trainable_variables

(0
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
}{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/time_distributed/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/time_distributed/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/time_distributed_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/time_distributed_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/lstm_8/lstm_cell_10/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/lstm_8/lstm_cell_10/recurrent_kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/lstm_8/lstm_cell_10/bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/lstm_9/lstm_cell_11/kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/lstm_9/lstm_cell_11/recurrent_kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/lstm_9/lstm_cell_11/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/time_distributed/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/time_distributed/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/time_distributed_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/time_distributed_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/lstm_8/lstm_cell_10/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/lstm_8/lstm_cell_10/recurrent_kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/lstm_8/lstm_cell_10/bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/lstm_9/lstm_cell_11/kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/lstm_9/lstm_cell_11/recurrent_kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/lstm_9/lstm_cell_11/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
&serving_default_time_distributed_inputPlaceholder*8
_output_shapes&
$:"??????????????????
*
dtype0*-
shape$:"??????????????????

?
StatefulPartitionedCallStatefulPartitionedCall&serving_default_time_distributed_inputtime_distributed/kerneltime_distributed/biastime_distributed_1/kerneltime_distributed_1/biaslstm_8/lstm_cell_10/kernel$lstm_8/lstm_cell_10/recurrent_kernellstm_8/lstm_cell_10/biaslstm_9/lstm_cell_11/kernel$lstm_9/lstm_cell_11/recurrent_kernellstm_9/lstm_cell_11/biasdense_9/kerneldense_9/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_238581
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp+time_distributed/kernel/Read/ReadVariableOp)time_distributed/bias/Read/ReadVariableOp-time_distributed_1/kernel/Read/ReadVariableOp+time_distributed_1/bias/Read/ReadVariableOp.lstm_8/lstm_cell_10/kernel/Read/ReadVariableOp8lstm_8/lstm_cell_10/recurrent_kernel/Read/ReadVariableOp,lstm_8/lstm_cell_10/bias/Read/ReadVariableOp.lstm_9/lstm_cell_11/kernel/Read/ReadVariableOp8lstm_9/lstm_cell_11/recurrent_kernel/Read/ReadVariableOp,lstm_9/lstm_cell_11/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp2Adam/time_distributed/kernel/m/Read/ReadVariableOp0Adam/time_distributed/bias/m/Read/ReadVariableOp4Adam/time_distributed_1/kernel/m/Read/ReadVariableOp2Adam/time_distributed_1/bias/m/Read/ReadVariableOp5Adam/lstm_8/lstm_cell_10/kernel/m/Read/ReadVariableOp?Adam/lstm_8/lstm_cell_10/recurrent_kernel/m/Read/ReadVariableOp3Adam/lstm_8/lstm_cell_10/bias/m/Read/ReadVariableOp5Adam/lstm_9/lstm_cell_11/kernel/m/Read/ReadVariableOp?Adam/lstm_9/lstm_cell_11/recurrent_kernel/m/Read/ReadVariableOp3Adam/lstm_9/lstm_cell_11/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp2Adam/time_distributed/kernel/v/Read/ReadVariableOp0Adam/time_distributed/bias/v/Read/ReadVariableOp4Adam/time_distributed_1/kernel/v/Read/ReadVariableOp2Adam/time_distributed_1/bias/v/Read/ReadVariableOp5Adam/lstm_8/lstm_cell_10/kernel/v/Read/ReadVariableOp?Adam/lstm_8/lstm_cell_10/recurrent_kernel/v/Read/ReadVariableOp3Adam/lstm_8/lstm_cell_10/bias/v/Read/ReadVariableOp5Adam/lstm_9/lstm_cell_11/kernel/v/Read/ReadVariableOp?Adam/lstm_9/lstm_cell_11/recurrent_kernel/v/Read/ReadVariableOp3Adam/lstm_9/lstm_cell_11/bias/v/Read/ReadVariableOpConst*:
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
__inference__traced_save_241417
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_9/kerneldense_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetime_distributed/kerneltime_distributed/biastime_distributed_1/kerneltime_distributed_1/biaslstm_8/lstm_cell_10/kernel$lstm_8/lstm_cell_10/recurrent_kernellstm_8/lstm_cell_10/biaslstm_9/lstm_cell_11/kernel$lstm_9/lstm_cell_11/recurrent_kernellstm_9/lstm_cell_11/biastotalcounttotal_1count_1Adam/dense_9/kernel/mAdam/dense_9/bias/mAdam/time_distributed/kernel/mAdam/time_distributed/bias/m Adam/time_distributed_1/kernel/mAdam/time_distributed_1/bias/m!Adam/lstm_8/lstm_cell_10/kernel/m+Adam/lstm_8/lstm_cell_10/recurrent_kernel/mAdam/lstm_8/lstm_cell_10/bias/m!Adam/lstm_9/lstm_cell_11/kernel/m+Adam/lstm_9/lstm_cell_11/recurrent_kernel/mAdam/lstm_9/lstm_cell_11/bias/mAdam/dense_9/kernel/vAdam/dense_9/bias/vAdam/time_distributed/kernel/vAdam/time_distributed/bias/v Adam/time_distributed_1/kernel/vAdam/time_distributed_1/bias/v!Adam/lstm_8/lstm_cell_10/kernel/v+Adam/lstm_8/lstm_cell_10/recurrent_kernel/vAdam/lstm_8/lstm_cell_10/bias/v!Adam/lstm_9/lstm_cell_11/kernel/v+Adam/lstm_9/lstm_cell_11/recurrent_kernel/vAdam/lstm_9/lstm_cell_11/bias/v*9
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
"__inference__traced_restore_241562??)
?D
?
B__inference_lstm_9_layer_call_and_return_conditional_losses_237436

inputs
lstm_cell_11_237354
lstm_cell_11_237356
lstm_cell_11_237358
identity??$lstm_cell_11/StatefulPartitionedCall?whileD
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
$lstm_cell_11/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_11_237354lstm_cell_11_237356lstm_cell_11_237358*
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
GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_11_layer_call_and_return_conditional_losses_2370402&
$lstm_cell_11/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_11_237354lstm_cell_11_237356lstm_cell_11_237358*
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
while_body_237367*
condR
while_cond_237366*K
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
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_11/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::2L
$lstm_cell_11/StatefulPartitionedCall$lstm_cell_11/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????2
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_239527

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
?W
?
B__inference_lstm_9_layer_call_and_return_conditional_losses_240804

inputs/
+lstm_cell_11_matmul_readvariableop_resource1
-lstm_cell_11_matmul_1_readvariableop_resource0
,lstm_cell_11_biasadd_readvariableop_resource
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
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOp?
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/MatMul?
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOp?
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/MatMul_1?
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/add?
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOp?
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/BiasAddj
lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/Const~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dim?
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_11/split?
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid?
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid_1?
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul}
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Relu?
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul_1?
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/add_1?
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid_2|
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Relu_1?
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
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
while_body_240719*
condR
while_cond_240718*K
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
?
B__inference_lstm_9_layer_call_and_return_conditional_losses_240476
inputs_0/
+lstm_cell_11_matmul_readvariableop_resource1
-lstm_cell_11_matmul_1_readvariableop_resource0
,lstm_cell_11_biasadd_readvariableop_resource
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
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOp?
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/MatMul?
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOp?
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/MatMul_1?
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/add?
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOp?
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/BiasAddj
lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/Const~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dim?
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_11/split?
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid?
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid_1?
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul}
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Relu?
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul_1?
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/add_1?
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid_2|
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Relu_1?
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
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
while_body_240391*
condR
while_cond_240390*K
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
?
}
(__inference_dense_9_layer_call_fn_240998

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
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_2383372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
O
3__inference_time_distributed_3_layer_call_fn_239662

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
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_2363332
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
?
?
while_cond_237723
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_237723___redundant_placeholder04
0while_while_cond_237723___redundant_placeholder14
0while_while_cond_237723___redundant_placeholder24
0while_while_cond_237723___redundant_placeholder3
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
?X
?
B__inference_lstm_8_layer_call_and_return_conditional_losses_239820
inputs_0/
+lstm_cell_10_matmul_readvariableop_resource1
-lstm_cell_10_matmul_1_readvariableop_resource0
,lstm_cell_10_biasadd_readvariableop_resource
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOp?
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/MatMul?
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOp?
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/MatMul_1?
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/add?
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOp?
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/BiasAddj
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim?
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_10/split?
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid?
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid_1?
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul}
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Relu?
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul_1?
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/add_1?
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Relu_1?
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
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
while_body_239735*
condR
while_cond_239734*K
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
while_body_239735
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_10_matmul_readvariableop_resource_09
5while_lstm_cell_10_matmul_1_readvariableop_resource_08
4while_lstm_cell_10_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_10_matmul_readvariableop_resource7
3while_lstm_cell_10_matmul_1_readvariableop_resource6
2while_lstm_cell_10_biasadd_readvariableop_resource??
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp?
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/MatMul?
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOp?
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/MatMul_1?
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/add?
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOp?
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/BiasAddv
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const?
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim?
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_10/split?
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid?
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid_1?
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul?
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Relu?
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul_1?
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/add_1?
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid_2?
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Relu_1?
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_2:z:0*
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
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
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
?
?
-__inference_lstm_cell_11_layer_call_fn_241242

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
GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_11_layer_call_and_return_conditional_losses_2370402
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
?
?
while_cond_238211
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_238211___redundant_placeholder04
0while_while_cond_238211___redundant_placeholder14
0while_while_cond_238211___redundant_placeholder24
0while_while_cond_238211___redundant_placeholder3
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
?
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_239613

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
?
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_239640

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
while_cond_240215
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_240215___redundant_placeholder04
0while_while_cond_240215___redundant_placeholder14
0while_while_cond_240215___redundant_placeholder24
0while_while_cond_240215___redundant_placeholder3
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
while_cond_236888
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_236888___redundant_placeholder04
0while_while_cond_236888___redundant_placeholder14
0while_while_cond_236888___redundant_placeholder24
0while_while_cond_236888___redundant_placeholder3
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
B__inference_lstm_9_layer_call_and_return_conditional_losses_238144

inputs/
+lstm_cell_11_matmul_readvariableop_resource1
-lstm_cell_11_matmul_1_readvariableop_resource0
,lstm_cell_11_biasadd_readvariableop_resource
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
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOp?
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/MatMul?
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOp?
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/MatMul_1?
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/add?
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOp?
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/BiasAddj
lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/Const~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dim?
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_11/split?
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid?
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid_1?
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul}
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Relu?
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul_1?
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/add_1?
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid_2|
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Relu_1?
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
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
while_body_238059*
condR
while_cond_238058*K
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
B__inference_lstm_8_layer_call_and_return_conditional_losses_237809

inputs/
+lstm_cell_10_matmul_readvariableop_resource1
-lstm_cell_10_matmul_1_readvariableop_resource0
,lstm_cell_10_biasadd_readvariableop_resource
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOp?
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/MatMul?
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOp?
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/MatMul_1?
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/add?
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOp?
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/BiasAddj
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim?
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_10/split?
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid?
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid_1?
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul}
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Relu?
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul_1?
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/add_1?
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Relu_1?
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
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
while_body_237724*
condR
while_cond_237723*K
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
?
?
D__inference_conv1d_2_layer_call_and_return_conditional_losses_235938

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
?W
?
B__inference_lstm_9_layer_call_and_return_conditional_losses_238297

inputs/
+lstm_cell_11_matmul_readvariableop_resource1
-lstm_cell_11_matmul_1_readvariableop_resource0
,lstm_cell_11_biasadd_readvariableop_resource
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
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOp?
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/MatMul?
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOp?
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/MatMul_1?
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/add?
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOp?
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/BiasAddj
lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/Const~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dim?
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_11/split?
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid?
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid_1?
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul}
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Relu?
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul_1?
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/add_1?
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid_2|
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Relu_1?
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
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
while_body_238212*
condR
while_cond_238211*K
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
?9
?
while_body_237877
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_10_matmul_readvariableop_resource_09
5while_lstm_cell_10_matmul_1_readvariableop_resource_08
4while_lstm_cell_10_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_10_matmul_readvariableop_resource7
3while_lstm_cell_10_matmul_1_readvariableop_resource6
2while_lstm_cell_10_biasadd_readvariableop_resource??
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp?
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/MatMul?
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOp?
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/MatMul_1?
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/add?
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOp?
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/BiasAddv
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const?
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim?
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_10/split?
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid?
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid_1?
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul?
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Relu?
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul_1?
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/add_1?
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid_2?
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Relu_1?
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_2:z:0*
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
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
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
?	
?
lstm_9_while_cond_238882*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3,
(lstm_9_while_less_lstm_9_strided_slice_1B
>lstm_9_while_lstm_9_while_cond_238882___redundant_placeholder0B
>lstm_9_while_lstm_9_while_cond_238882___redundant_placeholder1B
>lstm_9_while_lstm_9_while_cond_238882___redundant_placeholder2B
>lstm_9_while_lstm_9_while_cond_238882___redundant_placeholder3
lstm_9_while_identity
?
lstm_9/while/LessLesslstm_9_while_placeholder(lstm_9_while_less_lstm_9_strided_slice_1*
T0*
_output_shapes
: 2
lstm_9/while/Lessr
lstm_9/while/IdentityIdentitylstm_9/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_9/while/Identity"7
lstm_9_while_identitylstm_9/while/Identity:output:0*S
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
H__inference_lstm_cell_11_layer_call_and_return_conditional_losses_237040

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
?
O
3__inference_time_distributed_2_layer_call_fn_239623

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
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_2362652
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
?S
?
%sequential_5_lstm_8_while_body_235671D
@sequential_5_lstm_8_while_sequential_5_lstm_8_while_loop_counterJ
Fsequential_5_lstm_8_while_sequential_5_lstm_8_while_maximum_iterations)
%sequential_5_lstm_8_while_placeholder+
'sequential_5_lstm_8_while_placeholder_1+
'sequential_5_lstm_8_while_placeholder_2+
'sequential_5_lstm_8_while_placeholder_3C
?sequential_5_lstm_8_while_sequential_5_lstm_8_strided_slice_1_0
{sequential_5_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_8_tensorarrayunstack_tensorlistfromtensor_0K
Gsequential_5_lstm_8_while_lstm_cell_10_matmul_readvariableop_resource_0M
Isequential_5_lstm_8_while_lstm_cell_10_matmul_1_readvariableop_resource_0L
Hsequential_5_lstm_8_while_lstm_cell_10_biasadd_readvariableop_resource_0&
"sequential_5_lstm_8_while_identity(
$sequential_5_lstm_8_while_identity_1(
$sequential_5_lstm_8_while_identity_2(
$sequential_5_lstm_8_while_identity_3(
$sequential_5_lstm_8_while_identity_4(
$sequential_5_lstm_8_while_identity_5A
=sequential_5_lstm_8_while_sequential_5_lstm_8_strided_slice_1}
ysequential_5_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_8_tensorarrayunstack_tensorlistfromtensorI
Esequential_5_lstm_8_while_lstm_cell_10_matmul_readvariableop_resourceK
Gsequential_5_lstm_8_while_lstm_cell_10_matmul_1_readvariableop_resourceJ
Fsequential_5_lstm_8_while_lstm_cell_10_biasadd_readvariableop_resource??
Ksequential_5/lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2M
Ksequential_5/lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=sequential_5/lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_5_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_8_tensorarrayunstack_tensorlistfromtensor_0%sequential_5_lstm_8_while_placeholderTsequential_5/lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02?
=sequential_5/lstm_8/while/TensorArrayV2Read/TensorListGetItem?
<sequential_5/lstm_8/while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOpGsequential_5_lstm_8_while_lstm_cell_10_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02>
<sequential_5/lstm_8/while/lstm_cell_10/MatMul/ReadVariableOp?
-sequential_5/lstm_8/while/lstm_cell_10/MatMulMatMulDsequential_5/lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential_5/lstm_8/while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-sequential_5/lstm_8/while/lstm_cell_10/MatMul?
>sequential_5/lstm_8/while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOpIsequential_5_lstm_8_while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02@
>sequential_5/lstm_8/while/lstm_cell_10/MatMul_1/ReadVariableOp?
/sequential_5/lstm_8/while/lstm_cell_10/MatMul_1MatMul'sequential_5_lstm_8_while_placeholder_2Fsequential_5/lstm_8/while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_5/lstm_8/while/lstm_cell_10/MatMul_1?
*sequential_5/lstm_8/while/lstm_cell_10/addAddV27sequential_5/lstm_8/while/lstm_cell_10/MatMul:product:09sequential_5/lstm_8/while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2,
*sequential_5/lstm_8/while/lstm_cell_10/add?
=sequential_5/lstm_8/while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOpHsequential_5_lstm_8_while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02?
=sequential_5/lstm_8/while/lstm_cell_10/BiasAdd/ReadVariableOp?
.sequential_5/lstm_8/while/lstm_cell_10/BiasAddBiasAdd.sequential_5/lstm_8/while/lstm_cell_10/add:z:0Esequential_5/lstm_8/while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.sequential_5/lstm_8/while/lstm_cell_10/BiasAdd?
,sequential_5/lstm_8/while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_5/lstm_8/while/lstm_cell_10/Const?
6sequential_5/lstm_8/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential_5/lstm_8/while/lstm_cell_10/split/split_dim?
,sequential_5/lstm_8/while/lstm_cell_10/splitSplit?sequential_5/lstm_8/while/lstm_cell_10/split/split_dim:output:07sequential_5/lstm_8/while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2.
,sequential_5/lstm_8/while/lstm_cell_10/split?
.sequential_5/lstm_8/while/lstm_cell_10/SigmoidSigmoid5sequential_5/lstm_8/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????220
.sequential_5/lstm_8/while/lstm_cell_10/Sigmoid?
0sequential_5/lstm_8/while/lstm_cell_10/Sigmoid_1Sigmoid5sequential_5/lstm_8/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????222
0sequential_5/lstm_8/while/lstm_cell_10/Sigmoid_1?
*sequential_5/lstm_8/while/lstm_cell_10/mulMul4sequential_5/lstm_8/while/lstm_cell_10/Sigmoid_1:y:0'sequential_5_lstm_8_while_placeholder_3*
T0*'
_output_shapes
:?????????22,
*sequential_5/lstm_8/while/lstm_cell_10/mul?
+sequential_5/lstm_8/while/lstm_cell_10/ReluRelu5sequential_5/lstm_8/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22-
+sequential_5/lstm_8/while/lstm_cell_10/Relu?
,sequential_5/lstm_8/while/lstm_cell_10/mul_1Mul2sequential_5/lstm_8/while/lstm_cell_10/Sigmoid:y:09sequential_5/lstm_8/while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22.
,sequential_5/lstm_8/while/lstm_cell_10/mul_1?
,sequential_5/lstm_8/while/lstm_cell_10/add_1AddV2.sequential_5/lstm_8/while/lstm_cell_10/mul:z:00sequential_5/lstm_8/while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22.
,sequential_5/lstm_8/while/lstm_cell_10/add_1?
0sequential_5/lstm_8/while/lstm_cell_10/Sigmoid_2Sigmoid5sequential_5/lstm_8/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????222
0sequential_5/lstm_8/while/lstm_cell_10/Sigmoid_2?
-sequential_5/lstm_8/while/lstm_cell_10/Relu_1Relu0sequential_5/lstm_8/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22/
-sequential_5/lstm_8/while/lstm_cell_10/Relu_1?
,sequential_5/lstm_8/while/lstm_cell_10/mul_2Mul4sequential_5/lstm_8/while/lstm_cell_10/Sigmoid_2:y:0;sequential_5/lstm_8/while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22.
,sequential_5/lstm_8/while/lstm_cell_10/mul_2?
>sequential_5/lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_5_lstm_8_while_placeholder_1%sequential_5_lstm_8_while_placeholder0sequential_5/lstm_8/while/lstm_cell_10/mul_2:z:0*
_output_shapes
: *
element_dtype02@
>sequential_5/lstm_8/while/TensorArrayV2Write/TensorListSetItem?
sequential_5/lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_5/lstm_8/while/add/y?
sequential_5/lstm_8/while/addAddV2%sequential_5_lstm_8_while_placeholder(sequential_5/lstm_8/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_5/lstm_8/while/add?
!sequential_5/lstm_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_5/lstm_8/while/add_1/y?
sequential_5/lstm_8/while/add_1AddV2@sequential_5_lstm_8_while_sequential_5_lstm_8_while_loop_counter*sequential_5/lstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_5/lstm_8/while/add_1?
"sequential_5/lstm_8/while/IdentityIdentity#sequential_5/lstm_8/while/add_1:z:0*
T0*
_output_shapes
: 2$
"sequential_5/lstm_8/while/Identity?
$sequential_5/lstm_8/while/Identity_1IdentityFsequential_5_lstm_8_while_sequential_5_lstm_8_while_maximum_iterations*
T0*
_output_shapes
: 2&
$sequential_5/lstm_8/while/Identity_1?
$sequential_5/lstm_8/while/Identity_2Identity!sequential_5/lstm_8/while/add:z:0*
T0*
_output_shapes
: 2&
$sequential_5/lstm_8/while/Identity_2?
$sequential_5/lstm_8/while/Identity_3IdentityNsequential_5/lstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2&
$sequential_5/lstm_8/while/Identity_3?
$sequential_5/lstm_8/while/Identity_4Identity0sequential_5/lstm_8/while/lstm_cell_10/mul_2:z:0*
T0*'
_output_shapes
:?????????22&
$sequential_5/lstm_8/while/Identity_4?
$sequential_5/lstm_8/while/Identity_5Identity0sequential_5/lstm_8/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22&
$sequential_5/lstm_8/while/Identity_5"Q
"sequential_5_lstm_8_while_identity+sequential_5/lstm_8/while/Identity:output:0"U
$sequential_5_lstm_8_while_identity_1-sequential_5/lstm_8/while/Identity_1:output:0"U
$sequential_5_lstm_8_while_identity_2-sequential_5/lstm_8/while/Identity_2:output:0"U
$sequential_5_lstm_8_while_identity_3-sequential_5/lstm_8/while/Identity_3:output:0"U
$sequential_5_lstm_8_while_identity_4-sequential_5/lstm_8/while/Identity_4:output:0"U
$sequential_5_lstm_8_while_identity_5-sequential_5/lstm_8/while/Identity_5:output:0"?
Fsequential_5_lstm_8_while_lstm_cell_10_biasadd_readvariableop_resourceHsequential_5_lstm_8_while_lstm_cell_10_biasadd_readvariableop_resource_0"?
Gsequential_5_lstm_8_while_lstm_cell_10_matmul_1_readvariableop_resourceIsequential_5_lstm_8_while_lstm_cell_10_matmul_1_readvariableop_resource_0"?
Esequential_5_lstm_8_while_lstm_cell_10_matmul_readvariableop_resourceGsequential_5_lstm_8_while_lstm_cell_10_matmul_readvariableop_resource_0"?
=sequential_5_lstm_8_while_sequential_5_lstm_8_strided_slice_1?sequential_5_lstm_8_while_sequential_5_lstm_8_strided_slice_1_0"?
ysequential_5_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_8_tensorarrayunstack_tensorlistfromtensor{sequential_5_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*Q
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
??
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_239367

inputsI
Etime_distributed_conv1d_2_conv1d_expanddims_1_readvariableop_resource=
9time_distributed_conv1d_2_biasadd_readvariableop_resourceK
Gtime_distributed_1_conv1d_3_conv1d_expanddims_1_readvariableop_resource?
;time_distributed_1_conv1d_3_biasadd_readvariableop_resource6
2lstm_8_lstm_cell_10_matmul_readvariableop_resource8
4lstm_8_lstm_cell_10_matmul_1_readvariableop_resource7
3lstm_8_lstm_cell_10_biasadd_readvariableop_resource6
2lstm_9_lstm_cell_11_matmul_readvariableop_resource8
4lstm_9_lstm_cell_11_matmul_1_readvariableop_resource7
3lstm_9_lstm_cell_11_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identity??lstm_8/while?lstm_9/whilef
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
lstm_8/ShapeShape%time_distributed_3/Reshape_1:output:0*
T0*
_output_shapes
:2
lstm_8/Shape?
lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice/stack?
lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_1?
lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_2?
lstm_8/strided_sliceStridedSlicelstm_8/Shape:output:0#lstm_8/strided_slice/stack:output:0%lstm_8/strided_slice/stack_1:output:0%lstm_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slicej
lstm_8/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_8/zeros/mul/y?
lstm_8/zeros/mulMullstm_8/strided_slice:output:0lstm_8/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/mulm
lstm_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_8/zeros/Less/y?
lstm_8/zeros/LessLesslstm_8/zeros/mul:z:0lstm_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/Lessp
lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_8/zeros/packed/1?
lstm_8/zeros/packedPacklstm_8/strided_slice:output:0lstm_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_8/zeros/packedm
lstm_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/zeros/Const?
lstm_8/zerosFilllstm_8/zeros/packed:output:0lstm_8/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_8/zerosn
lstm_8/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_8/zeros_1/mul/y?
lstm_8/zeros_1/mulMullstm_8/strided_slice:output:0lstm_8/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/mulq
lstm_8/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_8/zeros_1/Less/y?
lstm_8/zeros_1/LessLesslstm_8/zeros_1/mul:z:0lstm_8/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/Lesst
lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_8/zeros_1/packed/1?
lstm_8/zeros_1/packedPacklstm_8/strided_slice:output:0 lstm_8/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_8/zeros_1/packedq
lstm_8/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/zeros_1/Const?
lstm_8/zeros_1Filllstm_8/zeros_1/packed:output:0lstm_8/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_8/zeros_1?
lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose/perm?
lstm_8/transpose	Transpose%time_distributed_3/Reshape_1:output:0lstm_8/transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
lstm_8/transposed
lstm_8/Shape_1Shapelstm_8/transpose:y:0*
T0*
_output_shapes
:2
lstm_8/Shape_1?
lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_1/stack?
lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_1?
lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_2?
lstm_8/strided_slice_1StridedSlicelstm_8/Shape_1:output:0%lstm_8/strided_slice_1/stack:output:0'lstm_8/strided_slice_1/stack_1:output:0'lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slice_1?
"lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_8/TensorArrayV2/element_shape?
lstm_8/TensorArrayV2TensorListReserve+lstm_8/TensorArrayV2/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2?
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2>
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_8/transpose:y:0Elstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_8/TensorArrayUnstack/TensorListFromTensor?
lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_2/stack?
lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_1?
lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_2?
lstm_8/strided_slice_2StridedSlicelstm_8/transpose:y:0%lstm_8/strided_slice_2/stack:output:0'lstm_8/strided_slice_2/stack_1:output:0'lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_8/strided_slice_2?
)lstm_8/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp2lstm_8_lstm_cell_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)lstm_8/lstm_cell_10/MatMul/ReadVariableOp?
lstm_8/lstm_cell_10/MatMulMatMullstm_8/strided_slice_2:output:01lstm_8/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_10/MatMul?
+lstm_8/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp4lstm_8_lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02-
+lstm_8/lstm_cell_10/MatMul_1/ReadVariableOp?
lstm_8/lstm_cell_10/MatMul_1MatMullstm_8/zeros:output:03lstm_8/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_10/MatMul_1?
lstm_8/lstm_cell_10/addAddV2$lstm_8/lstm_cell_10/MatMul:product:0&lstm_8/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_10/add?
*lstm_8/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp3lstm_8_lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*lstm_8/lstm_cell_10/BiasAdd/ReadVariableOp?
lstm_8/lstm_cell_10/BiasAddBiasAddlstm_8/lstm_cell_10/add:z:02lstm_8/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_10/BiasAddx
lstm_8/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/lstm_cell_10/Const?
#lstm_8/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#lstm_8/lstm_cell_10/split/split_dim?
lstm_8/lstm_cell_10/splitSplit,lstm_8/lstm_cell_10/split/split_dim:output:0$lstm_8/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_8/lstm_cell_10/split?
lstm_8/lstm_cell_10/SigmoidSigmoid"lstm_8/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/Sigmoid?
lstm_8/lstm_cell_10/Sigmoid_1Sigmoid"lstm_8/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/Sigmoid_1?
lstm_8/lstm_cell_10/mulMul!lstm_8/lstm_cell_10/Sigmoid_1:y:0lstm_8/zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/mul?
lstm_8/lstm_cell_10/ReluRelu"lstm_8/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/Relu?
lstm_8/lstm_cell_10/mul_1Mullstm_8/lstm_cell_10/Sigmoid:y:0&lstm_8/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/mul_1?
lstm_8/lstm_cell_10/add_1AddV2lstm_8/lstm_cell_10/mul:z:0lstm_8/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/add_1?
lstm_8/lstm_cell_10/Sigmoid_2Sigmoid"lstm_8/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/Sigmoid_2?
lstm_8/lstm_cell_10/Relu_1Relulstm_8/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/Relu_1?
lstm_8/lstm_cell_10/mul_2Mul!lstm_8/lstm_cell_10/Sigmoid_2:y:0(lstm_8/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/mul_2?
$lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2&
$lstm_8/TensorArrayV2_1/element_shape?
lstm_8/TensorArrayV2_1TensorListReserve-lstm_8/TensorArrayV2_1/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2_1\
lstm_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/time?
lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_8/while/maximum_iterationsx
lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/while/loop_counter?
lstm_8/whileWhile"lstm_8/while/loop_counter:output:0(lstm_8/while/maximum_iterations:output:0lstm_8/time:output:0lstm_8/TensorArrayV2_1:handle:0lstm_8/zeros:output:0lstm_8/zeros_1:output:0lstm_8/strided_slice_1:output:0>lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_8_lstm_cell_10_matmul_readvariableop_resource4lstm_8_lstm_cell_10_matmul_1_readvariableop_resource3lstm_8_lstm_cell_10_biasadd_readvariableop_resource*
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
lstm_8_while_body_239127*$
condR
lstm_8_while_cond_239126*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
lstm_8/while?
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_8/TensorArrayV2Stack/TensorListStackTensorListStacklstm_8/while:output:3@lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02+
)lstm_8/TensorArrayV2Stack/TensorListStack?
lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_8/strided_slice_3/stack?
lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_8/strided_slice_3/stack_1?
lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_3/stack_2?
lstm_8/strided_slice_3StridedSlice2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_8/strided_slice_3/stack:output:0'lstm_8/strided_slice_3/stack_1:output:0'lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_8/strided_slice_3?
lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose_1/perm?
lstm_8/transpose_1	Transpose2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_8/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
lstm_8/transpose_1t
lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/runtimeb
lstm_9/ShapeShapelstm_8/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_9/Shape?
lstm_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice/stack?
lstm_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_9/strided_slice/stack_1?
lstm_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_9/strided_slice/stack_2?
lstm_9/strided_sliceStridedSlicelstm_9/Shape:output:0#lstm_9/strided_slice/stack:output:0%lstm_9/strided_slice/stack_1:output:0%lstm_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_9/strided_slicej
lstm_9/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/zeros/mul/y?
lstm_9/zeros/mulMullstm_9/strided_slice:output:0lstm_9/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros/mulm
lstm_9/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_9/zeros/Less/y?
lstm_9/zeros/LessLesslstm_9/zeros/mul:z:0lstm_9/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros/Lessp
lstm_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/zeros/packed/1?
lstm_9/zeros/packedPacklstm_9/strided_slice:output:0lstm_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_9/zeros/packedm
lstm_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/zeros/Const?
lstm_9/zerosFilllstm_9/zeros/packed:output:0lstm_9/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_9/zerosn
lstm_9/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/zeros_1/mul/y?
lstm_9/zeros_1/mulMullstm_9/strided_slice:output:0lstm_9/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros_1/mulq
lstm_9/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_9/zeros_1/Less/y?
lstm_9/zeros_1/LessLesslstm_9/zeros_1/mul:z:0lstm_9/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros_1/Lesst
lstm_9/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/zeros_1/packed/1?
lstm_9/zeros_1/packedPacklstm_9/strided_slice:output:0 lstm_9/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_9/zeros_1/packedq
lstm_9/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/zeros_1/Const?
lstm_9/zeros_1Filllstm_9/zeros_1/packed:output:0lstm_9/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_9/zeros_1?
lstm_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_9/transpose/perm?
lstm_9/transpose	Transposelstm_8/transpose_1:y:0lstm_9/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
lstm_9/transposed
lstm_9/Shape_1Shapelstm_9/transpose:y:0*
T0*
_output_shapes
:2
lstm_9/Shape_1?
lstm_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice_1/stack?
lstm_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_1/stack_1?
lstm_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_1/stack_2?
lstm_9/strided_slice_1StridedSlicelstm_9/Shape_1:output:0%lstm_9/strided_slice_1/stack:output:0'lstm_9/strided_slice_1/stack_1:output:0'lstm_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_9/strided_slice_1?
"lstm_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_9/TensorArrayV2/element_shape?
lstm_9/TensorArrayV2TensorListReserve+lstm_9/TensorArrayV2/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_9/TensorArrayV2?
<lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2>
<lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_9/transpose:y:0Elstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_9/TensorArrayUnstack/TensorListFromTensor?
lstm_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice_2/stack?
lstm_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_2/stack_1?
lstm_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_2/stack_2?
lstm_9/strided_slice_2StridedSlicelstm_9/transpose:y:0%lstm_9/strided_slice_2/stack:output:0'lstm_9/strided_slice_2/stack_1:output:0'lstm_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_9/strided_slice_2?
)lstm_9/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp2lstm_9_lstm_cell_11_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02+
)lstm_9/lstm_cell_11/MatMul/ReadVariableOp?
lstm_9/lstm_cell_11/MatMulMatMullstm_9/strided_slice_2:output:01lstm_9/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_9/lstm_cell_11/MatMul?
+lstm_9/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp4lstm_9_lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02-
+lstm_9/lstm_cell_11/MatMul_1/ReadVariableOp?
lstm_9/lstm_cell_11/MatMul_1MatMullstm_9/zeros:output:03lstm_9/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_9/lstm_cell_11/MatMul_1?
lstm_9/lstm_cell_11/addAddV2$lstm_9/lstm_cell_11/MatMul:product:0&lstm_9/lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_9/lstm_cell_11/add?
*lstm_9/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp3lstm_9_lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02,
*lstm_9/lstm_cell_11/BiasAdd/ReadVariableOp?
lstm_9/lstm_cell_11/BiasAddBiasAddlstm_9/lstm_cell_11/add:z:02lstm_9/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_9/lstm_cell_11/BiasAddx
lstm_9/lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/lstm_cell_11/Const?
#lstm_9/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#lstm_9/lstm_cell_11/split/split_dim?
lstm_9/lstm_cell_11/splitSplit,lstm_9/lstm_cell_11/split/split_dim:output:0$lstm_9/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_9/lstm_cell_11/split?
lstm_9/lstm_cell_11/SigmoidSigmoid"lstm_9/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/Sigmoid?
lstm_9/lstm_cell_11/Sigmoid_1Sigmoid"lstm_9/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/Sigmoid_1?
lstm_9/lstm_cell_11/mulMul!lstm_9/lstm_cell_11/Sigmoid_1:y:0lstm_9/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/mul?
lstm_9/lstm_cell_11/ReluRelu"lstm_9/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/Relu?
lstm_9/lstm_cell_11/mul_1Mullstm_9/lstm_cell_11/Sigmoid:y:0&lstm_9/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/mul_1?
lstm_9/lstm_cell_11/add_1AddV2lstm_9/lstm_cell_11/mul:z:0lstm_9/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/add_1?
lstm_9/lstm_cell_11/Sigmoid_2Sigmoid"lstm_9/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/Sigmoid_2?
lstm_9/lstm_cell_11/Relu_1Relulstm_9/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/Relu_1?
lstm_9/lstm_cell_11/mul_2Mul!lstm_9/lstm_cell_11/Sigmoid_2:y:0(lstm_9/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/mul_2?
$lstm_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$lstm_9/TensorArrayV2_1/element_shape?
lstm_9/TensorArrayV2_1TensorListReserve-lstm_9/TensorArrayV2_1/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_9/TensorArrayV2_1\
lstm_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_9/time?
lstm_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_9/while/maximum_iterationsx
lstm_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_9/while/loop_counter?
lstm_9/whileWhile"lstm_9/while/loop_counter:output:0(lstm_9/while/maximum_iterations:output:0lstm_9/time:output:0lstm_9/TensorArrayV2_1:handle:0lstm_9/zeros:output:0lstm_9/zeros_1:output:0lstm_9/strided_slice_1:output:0>lstm_9/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_9_lstm_cell_11_matmul_readvariableop_resource4lstm_9_lstm_cell_11_matmul_1_readvariableop_resource3lstm_9_lstm_cell_11_biasadd_readvariableop_resource*
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
lstm_9_while_body_239276*$
condR
lstm_9_while_cond_239275*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
lstm_9/while?
7lstm_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7lstm_9/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_9/TensorArrayV2Stack/TensorListStackTensorListStacklstm_9/while:output:3@lstm_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02+
)lstm_9/TensorArrayV2Stack/TensorListStack?
lstm_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_9/strided_slice_3/stack?
lstm_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_9/strided_slice_3/stack_1?
lstm_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_3/stack_2?
lstm_9/strided_slice_3StridedSlice2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_9/strided_slice_3/stack:output:0'lstm_9/strided_slice_3/stack_1:output:0'lstm_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_9/strided_slice_3?
lstm_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_9/transpose_1/perm?
lstm_9/transpose_1	Transpose2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_9/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
lstm_9/transpose_1t
lstm_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/runtime?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMullstm_9/strided_slice_3:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense_9/BiasAdd?
IdentityIdentitydense_9/BiasAdd:output:0^lstm_8/while^lstm_9/while*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::2
lstm_8/whilelstm_8/while2
lstm_9/whilelstm_9/while:` \
8
_output_shapes&
$:"??????????????????

 
_user_specified_nameinputs
?
?
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_241092

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
g
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_236182

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
?
?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_239555

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
?9
?
while_body_237724
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_10_matmul_readvariableop_resource_09
5while_lstm_cell_10_matmul_1_readvariableop_resource_08
4while_lstm_cell_10_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_10_matmul_readvariableop_resource7
3while_lstm_cell_10_matmul_1_readvariableop_resource6
2while_lstm_cell_10_biasadd_readvariableop_resource??
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp?
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/MatMul?
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOp?
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/MatMul_1?
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/add?
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOp?
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/BiasAddv
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const?
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim?
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_10/split?
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid?
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid_1?
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul?
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Relu?
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul_1?
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/add_1?
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid_2?
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Relu_1?
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_2:z:0*
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
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
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
?
?
'__inference_lstm_8_layer_call_fn_239984
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
B__inference_lstm_8_layer_call_and_return_conditional_losses_2368262
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
?
?
H__inference_lstm_cell_11_layer_call_and_return_conditional_losses_237073

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
?
?
while_cond_236756
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_236756___redundant_placeholder04
0while_while_cond_236756___redundant_placeholder14
0while_while_cond_236756___redundant_placeholder24
0while_while_cond_236756___redundant_placeholder3
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
Ŀ
?
"__inference__traced_restore_241562
file_prefix#
assignvariableop_dense_9_kernel#
assignvariableop_1_dense_9_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate.
*assignvariableop_7_time_distributed_kernel,
(assignvariableop_8_time_distributed_bias0
,assignvariableop_9_time_distributed_1_kernel/
+assignvariableop_10_time_distributed_1_bias2
.assignvariableop_11_lstm_8_lstm_cell_10_kernel<
8assignvariableop_12_lstm_8_lstm_cell_10_recurrent_kernel0
,assignvariableop_13_lstm_8_lstm_cell_10_bias2
.assignvariableop_14_lstm_9_lstm_cell_11_kernel<
8assignvariableop_15_lstm_9_lstm_cell_11_recurrent_kernel0
,assignvariableop_16_lstm_9_lstm_cell_11_bias
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_1-
)assignvariableop_21_adam_dense_9_kernel_m+
'assignvariableop_22_adam_dense_9_bias_m6
2assignvariableop_23_adam_time_distributed_kernel_m4
0assignvariableop_24_adam_time_distributed_bias_m8
4assignvariableop_25_adam_time_distributed_1_kernel_m6
2assignvariableop_26_adam_time_distributed_1_bias_m9
5assignvariableop_27_adam_lstm_8_lstm_cell_10_kernel_mC
?assignvariableop_28_adam_lstm_8_lstm_cell_10_recurrent_kernel_m7
3assignvariableop_29_adam_lstm_8_lstm_cell_10_bias_m9
5assignvariableop_30_adam_lstm_9_lstm_cell_11_kernel_mC
?assignvariableop_31_adam_lstm_9_lstm_cell_11_recurrent_kernel_m7
3assignvariableop_32_adam_lstm_9_lstm_cell_11_bias_m-
)assignvariableop_33_adam_dense_9_kernel_v+
'assignvariableop_34_adam_dense_9_bias_v6
2assignvariableop_35_adam_time_distributed_kernel_v4
0assignvariableop_36_adam_time_distributed_bias_v8
4assignvariableop_37_adam_time_distributed_1_kernel_v6
2assignvariableop_38_adam_time_distributed_1_bias_v9
5assignvariableop_39_adam_lstm_8_lstm_cell_10_kernel_vC
?assignvariableop_40_adam_lstm_8_lstm_cell_10_recurrent_kernel_v7
3assignvariableop_41_adam_lstm_8_lstm_cell_10_bias_v9
5assignvariableop_42_adam_lstm_9_lstm_cell_11_kernel_vC
?assignvariableop_43_adam_lstm_9_lstm_cell_11_recurrent_kernel_v7
3assignvariableop_44_adam_lstm_9_lstm_cell_11_bias_v
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOpAssignVariableOpassignvariableop_dense_9_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_9_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_11AssignVariableOp.assignvariableop_11_lstm_8_lstm_cell_10_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp8assignvariableop_12_lstm_8_lstm_cell_10_recurrent_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp,assignvariableop_13_lstm_8_lstm_cell_10_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp.assignvariableop_14_lstm_9_lstm_cell_11_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp8assignvariableop_15_lstm_9_lstm_cell_11_recurrent_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp,assignvariableop_16_lstm_9_lstm_cell_11_biasIdentity_16:output:0"/device:CPU:0*
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
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_9_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_9_bias_mIdentity_22:output:0"/device:CPU:0*
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
AssignVariableOp_27AssignVariableOp5assignvariableop_27_adam_lstm_8_lstm_cell_10_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp?assignvariableop_28_adam_lstm_8_lstm_cell_10_recurrent_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp3assignvariableop_29_adam_lstm_8_lstm_cell_10_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp5assignvariableop_30_adam_lstm_9_lstm_cell_11_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp?assignvariableop_31_adam_lstm_9_lstm_cell_11_recurrent_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp3assignvariableop_32_adam_lstm_9_lstm_cell_11_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_9_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_9_bias_vIdentity_34:output:0"/device:CPU:0*
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
AssignVariableOp_39AssignVariableOp5assignvariableop_39_adam_lstm_8_lstm_cell_10_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp?assignvariableop_40_adam_lstm_8_lstm_cell_10_recurrent_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp3assignvariableop_41_adam_lstm_8_lstm_cell_10_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_lstm_9_lstm_cell_11_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp?assignvariableop_43_adam_lstm_9_lstm_cell_11_recurrent_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp3assignvariableop_44_adam_lstm_9_lstm_cell_11_bias_vIdentity_44:output:0"/device:CPU:0*
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
?
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_236243

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
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_2361822!
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
?
~
)__inference_conv1d_3_layer_call_fn_241048

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
D__inference_conv1d_3_layer_call_and_return_conditional_losses_2360692
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
?9
?
while_body_238212
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_11_matmul_readvariableop_resource_09
5while_lstm_cell_11_matmul_1_readvariableop_resource_08
4while_lstm_cell_11_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_11_matmul_readvariableop_resource7
3while_lstm_cell_11_matmul_1_readvariableop_resource6
2while_lstm_cell_11_biasadd_readvariableop_resource??
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
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp?
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/MatMul?
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOp?
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/MatMul_1?
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/add?
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOp?
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/BiasAddv
while/lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_11/Const?
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dim?
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_11/split?
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid?
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid_1?
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul?
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Relu?
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul_1?
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/add_1?
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid_2?
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Relu_1?
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_11/mul_2:z:0*
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
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_11_biasadd_readvariableop_resource4while_lstm_cell_11_biasadd_readvariableop_resource_0"l
3while_lstm_cell_11_matmul_1_readvariableop_resource5while_lstm_cell_11_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_11_matmul_readvariableop_resource3while_lstm_cell_11_matmul_readvariableop_resource_0"0
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
while_cond_240062
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_240062___redundant_placeholder04
0while_while_cond_240062___redundant_placeholder14
0while_while_cond_240062___redundant_placeholder24
0while_while_cond_240062___redundant_placeholder3
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
?
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_236333

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
E__inference_flatten_1_layer_call_and_return_conditional_losses_2362852
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
?
?
'__inference_lstm_8_layer_call_fn_240323

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
B__inference_lstm_8_layer_call_and_return_conditional_losses_2379622
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
?%
?
while_body_237367
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_11_237391_0
while_lstm_cell_11_237393_0
while_lstm_cell_11_237395_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_11_237391
while_lstm_cell_11_237393
while_lstm_cell_11_237395??*while/lstm_cell_11/StatefulPartitionedCall?
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
*while/lstm_cell_11/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_11_237391_0while_lstm_cell_11_237393_0while_lstm_cell_11_237395_0*
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
GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_11_layer_call_and_return_conditional_losses_2370402,
*while/lstm_cell_11/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_11/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_11/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_11/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_11/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_11/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_11/StatefulPartitionedCall:output:1+^while/lstm_cell_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_11/StatefulPartitionedCall:output:2+^while/lstm_cell_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_11_237391while_lstm_cell_11_237391_0"8
while_lstm_cell_11_237393while_lstm_cell_11_237393_0"8
while_lstm_cell_11_237395while_lstm_cell_11_237395_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2X
*while/lstm_cell_11/StatefulPartitionedCall*while/lstm_cell_11/StatefulPartitionedCall: 
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
-__inference_lstm_cell_11_layer_call_fn_241259

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
GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_11_layer_call_and_return_conditional_losses_2370732
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
1__inference_time_distributed_layer_call_fn_239499

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
L__inference_time_distributed_layer_call_and_return_conditional_losses_2360352
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
?B
?
lstm_9_while_body_238883*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3)
%lstm_9_while_lstm_9_strided_slice_1_0e
alstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0>
:lstm_9_while_lstm_cell_11_matmul_readvariableop_resource_0@
<lstm_9_while_lstm_cell_11_matmul_1_readvariableop_resource_0?
;lstm_9_while_lstm_cell_11_biasadd_readvariableop_resource_0
lstm_9_while_identity
lstm_9_while_identity_1
lstm_9_while_identity_2
lstm_9_while_identity_3
lstm_9_while_identity_4
lstm_9_while_identity_5'
#lstm_9_while_lstm_9_strided_slice_1c
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor<
8lstm_9_while_lstm_cell_11_matmul_readvariableop_resource>
:lstm_9_while_lstm_cell_11_matmul_1_readvariableop_resource=
9lstm_9_while_lstm_cell_11_biasadd_readvariableop_resource??
>lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2@
>lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0lstm_9_while_placeholderGlstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype022
0lstm_9/while/TensorArrayV2Read/TensorListGetItem?
/lstm_9/while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp:lstm_9_while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype021
/lstm_9/while/lstm_cell_11/MatMul/ReadVariableOp?
 lstm_9/while/lstm_cell_11/MatMulMatMul7lstm_9/while/TensorArrayV2Read/TensorListGetItem:item:07lstm_9/while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2"
 lstm_9/while/lstm_cell_11/MatMul?
1lstm_9/while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp<lstm_9_while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype023
1lstm_9/while/lstm_cell_11/MatMul_1/ReadVariableOp?
"lstm_9/while/lstm_cell_11/MatMul_1MatMullstm_9_while_placeholder_29lstm_9/while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2$
"lstm_9/while/lstm_cell_11/MatMul_1?
lstm_9/while/lstm_cell_11/addAddV2*lstm_9/while/lstm_cell_11/MatMul:product:0,lstm_9/while/lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_9/while/lstm_cell_11/add?
0lstm_9/while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp;lstm_9_while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype022
0lstm_9/while/lstm_cell_11/BiasAdd/ReadVariableOp?
!lstm_9/while/lstm_cell_11/BiasAddBiasAdd!lstm_9/while/lstm_cell_11/add:z:08lstm_9/while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2#
!lstm_9/while/lstm_cell_11/BiasAdd?
lstm_9/while/lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
lstm_9/while/lstm_cell_11/Const?
)lstm_9/while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)lstm_9/while/lstm_cell_11/split/split_dim?
lstm_9/while/lstm_cell_11/splitSplit2lstm_9/while/lstm_cell_11/split/split_dim:output:0*lstm_9/while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2!
lstm_9/while/lstm_cell_11/split?
!lstm_9/while/lstm_cell_11/SigmoidSigmoid(lstm_9/while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_9/while/lstm_cell_11/Sigmoid?
#lstm_9/while/lstm_cell_11/Sigmoid_1Sigmoid(lstm_9/while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????2%
#lstm_9/while/lstm_cell_11/Sigmoid_1?
lstm_9/while/lstm_cell_11/mulMul'lstm_9/while/lstm_cell_11/Sigmoid_1:y:0lstm_9_while_placeholder_3*
T0*'
_output_shapes
:?????????2
lstm_9/while/lstm_cell_11/mul?
lstm_9/while/lstm_cell_11/ReluRelu(lstm_9/while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2 
lstm_9/while/lstm_cell_11/Relu?
lstm_9/while/lstm_cell_11/mul_1Mul%lstm_9/while/lstm_cell_11/Sigmoid:y:0,lstm_9/while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2!
lstm_9/while/lstm_cell_11/mul_1?
lstm_9/while/lstm_cell_11/add_1AddV2!lstm_9/while/lstm_cell_11/mul:z:0#lstm_9/while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2!
lstm_9/while/lstm_cell_11/add_1?
#lstm_9/while/lstm_cell_11/Sigmoid_2Sigmoid(lstm_9/while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????2%
#lstm_9/while/lstm_cell_11/Sigmoid_2?
 lstm_9/while/lstm_cell_11/Relu_1Relu#lstm_9/while/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2"
 lstm_9/while/lstm_cell_11/Relu_1?
lstm_9/while/lstm_cell_11/mul_2Mul'lstm_9/while/lstm_cell_11/Sigmoid_2:y:0.lstm_9/while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2!
lstm_9/while/lstm_cell_11/mul_2?
1lstm_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_9_while_placeholder_1lstm_9_while_placeholder#lstm_9/while/lstm_cell_11/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_9/while/TensorArrayV2Write/TensorListSetItemj
lstm_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/while/add/y?
lstm_9/while/addAddV2lstm_9_while_placeholderlstm_9/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_9/while/addn
lstm_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/while/add_1/y?
lstm_9/while/add_1AddV2&lstm_9_while_lstm_9_while_loop_counterlstm_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_9/while/add_1s
lstm_9/while/IdentityIdentitylstm_9/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_9/while/Identity?
lstm_9/while/Identity_1Identity,lstm_9_while_lstm_9_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_9/while/Identity_1u
lstm_9/while/Identity_2Identitylstm_9/while/add:z:0*
T0*
_output_shapes
: 2
lstm_9/while/Identity_2?
lstm_9/while/Identity_3IdentityAlstm_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_9/while/Identity_3?
lstm_9/while/Identity_4Identity#lstm_9/while/lstm_cell_11/mul_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_9/while/Identity_4?
lstm_9/while/Identity_5Identity#lstm_9/while/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_9/while/Identity_5"7
lstm_9_while_identitylstm_9/while/Identity:output:0";
lstm_9_while_identity_1 lstm_9/while/Identity_1:output:0";
lstm_9_while_identity_2 lstm_9/while/Identity_2:output:0";
lstm_9_while_identity_3 lstm_9/while/Identity_3:output:0";
lstm_9_while_identity_4 lstm_9/while/Identity_4:output:0";
lstm_9_while_identity_5 lstm_9/while/Identity_5:output:0"L
#lstm_9_while_lstm_9_strided_slice_1%lstm_9_while_lstm_9_strided_slice_1_0"x
9lstm_9_while_lstm_cell_11_biasadd_readvariableop_resource;lstm_9_while_lstm_cell_11_biasadd_readvariableop_resource_0"z
:lstm_9_while_lstm_cell_11_matmul_1_readvariableop_resource<lstm_9_while_lstm_cell_11_matmul_1_readvariableop_resource_0"v
8lstm_9_while_lstm_cell_11_matmul_readvariableop_resource:lstm_9_while_lstm_cell_11_matmul_readvariableop_resource_0"?
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensoralstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0*Q
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
?9
?
while_body_240719
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_11_matmul_readvariableop_resource_09
5while_lstm_cell_11_matmul_1_readvariableop_resource_08
4while_lstm_cell_11_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_11_matmul_readvariableop_resource7
3while_lstm_cell_11_matmul_1_readvariableop_resource6
2while_lstm_cell_11_biasadd_readvariableop_resource??
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
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp?
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/MatMul?
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOp?
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/MatMul_1?
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/add?
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOp?
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/BiasAddv
while/lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_11/Const?
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dim?
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_11/split?
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid?
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid_1?
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul?
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Relu?
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul_1?
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/add_1?
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid_2?
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Relu_1?
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_11/mul_2:z:0*
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
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_11_biasadd_readvariableop_resource4while_lstm_cell_11_biasadd_readvariableop_resource_0"l
3while_lstm_cell_11_matmul_1_readvariableop_resource5while_lstm_cell_11_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_11_matmul_readvariableop_resource3while_lstm_cell_11_matmul_readvariableop_resource_0"0
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
?D
?
B__inference_lstm_8_layer_call_and_return_conditional_losses_236826

inputs
lstm_cell_10_236744
lstm_cell_10_236746
lstm_cell_10_236748
identity??$lstm_cell_10/StatefulPartitionedCall?whileD
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
$lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10_236744lstm_cell_10_236746lstm_cell_10_236748*
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
GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_2364302&
$lstm_cell_10/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10_236744lstm_cell_10_236746lstm_cell_10_236748*
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
while_body_236757*
condR
while_cond_236756*K
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
IdentityIdentitytranspose_1:y:0%^lstm_cell_10/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2L
$lstm_cell_10/StatefulPartitionedCall$lstm_cell_10/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?0
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_238354
time_distributed_input
time_distributed_237599
time_distributed_237601
time_distributed_1_237624
time_distributed_1_237626
lstm_8_237985
lstm_8_237987
lstm_8_237989
lstm_9_238320
lstm_9_238322
lstm_9_238324
dense_9_238348
dense_9_238350
identity??dense_9/StatefulPartitionedCall?lstm_8/StatefulPartitionedCall?lstm_9/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCalltime_distributed_inputtime_distributed_237599time_distributed_237601*
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
L__inference_time_distributed_layer_call_and_return_conditional_losses_2360052*
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
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_237624time_distributed_1_237626*
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
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_2361362,
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
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_2362432$
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
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_2363332$
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
lstm_8/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0lstm_8_237985lstm_8_237987lstm_8_237989*
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
B__inference_lstm_8_layer_call_and_return_conditional_losses_2378092 
lstm_8/StatefulPartitionedCall?
lstm_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0lstm_9_238320lstm_9_238322lstm_9_238324*
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
B__inference_lstm_9_layer_call_and_return_conditional_losses_2381442 
lstm_9/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_9/StatefulPartitionedCall:output:0dense_9_238348dense_9_238350*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_2383372!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_9/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:p l
8
_output_shapes&
$:"??????????????????

0
_user_specified_nametime_distributed_input
?
?
while_cond_237366
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_237366___redundant_placeholder04
0while_while_cond_237366___redundant_placeholder14
0while_while_cond_237366___redundant_placeholder24
0while_while_cond_237366___redundant_placeholder3
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
?9
?
while_body_240391
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_11_matmul_readvariableop_resource_09
5while_lstm_cell_11_matmul_1_readvariableop_resource_08
4while_lstm_cell_11_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_11_matmul_readvariableop_resource7
3while_lstm_cell_11_matmul_1_readvariableop_resource6
2while_lstm_cell_11_biasadd_readvariableop_resource??
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
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp?
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/MatMul?
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOp?
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/MatMul_1?
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/add?
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOp?
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/BiasAddv
while/lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_11/Const?
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dim?
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_11/split?
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid?
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid_1?
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul?
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Relu?
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul_1?
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/add_1?
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid_2?
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Relu_1?
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_11/mul_2:z:0*
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
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_11_biasadd_readvariableop_resource4while_lstm_cell_11_biasadd_readvariableop_resource_0"l
3while_lstm_cell_11_matmul_1_readvariableop_resource5while_lstm_cell_11_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_11_matmul_readvariableop_resource3while_lstm_cell_11_matmul_readvariableop_resource_0"0
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
L__inference_time_distributed_layer_call_and_return_conditional_losses_239453

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
?
?
while_cond_240871
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_240871___redundant_placeholder04
0while_while_cond_240871___redundant_placeholder14
0while_while_cond_240871___redundant_placeholder24
0while_while_cond_240871___redundant_placeholder3
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
?
?
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_236430

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
?
-__inference_sequential_5_layer_call_fn_238470
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
:?????????(*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_2384432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

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
?W
?
B__inference_lstm_8_layer_call_and_return_conditional_losses_240301

inputs/
+lstm_cell_10_matmul_readvariableop_resource1
-lstm_cell_10_matmul_1_readvariableop_resource0
,lstm_cell_10_biasadd_readvariableop_resource
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOp?
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/MatMul?
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOp?
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/MatMul_1?
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/add?
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOp?
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/BiasAddj
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim?
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_10/split?
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid?
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid_1?
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul}
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Relu?
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul_1?
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/add_1?
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Relu_1?
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
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
while_body_240216*
condR
while_cond_240215*K
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
?
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_239657

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
?	
?
-__inference_sequential_5_layer_call_fn_238542
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
:?????????(*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_2385152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

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
?
?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_236136

inputs
conv1d_3_236125
conv1d_3_236127
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
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv1d_3_236125conv1d_3_236127*
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
D__inference_conv1d_3_layer_call_and_return_conditional_losses_2360692"
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
while_cond_239887
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_239887___redundant_placeholder04
0while_while_cond_239887___redundant_placeholder14
0while_while_cond_239887___redundant_placeholder24
0while_while_cond_239887___redundant_placeholder3
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
?
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_241054

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
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_239481

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
?W
?
B__inference_lstm_8_layer_call_and_return_conditional_losses_240148

inputs/
+lstm_cell_10_matmul_readvariableop_resource1
-lstm_cell_10_matmul_1_readvariableop_resource0
,lstm_cell_10_biasadd_readvariableop_resource
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOp?
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/MatMul?
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOp?
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/MatMul_1?
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/add?
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOp?
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/BiasAddj
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim?
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_10/split?
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid?
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid_1?
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul}
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Relu?
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul_1?
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/add_1?
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Relu_1?
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
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
while_body_240063*
condR
while_cond_240062*K
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
?
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_236285

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
?B
?
lstm_9_while_body_239276*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3)
%lstm_9_while_lstm_9_strided_slice_1_0e
alstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0>
:lstm_9_while_lstm_cell_11_matmul_readvariableop_resource_0@
<lstm_9_while_lstm_cell_11_matmul_1_readvariableop_resource_0?
;lstm_9_while_lstm_cell_11_biasadd_readvariableop_resource_0
lstm_9_while_identity
lstm_9_while_identity_1
lstm_9_while_identity_2
lstm_9_while_identity_3
lstm_9_while_identity_4
lstm_9_while_identity_5'
#lstm_9_while_lstm_9_strided_slice_1c
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor<
8lstm_9_while_lstm_cell_11_matmul_readvariableop_resource>
:lstm_9_while_lstm_cell_11_matmul_1_readvariableop_resource=
9lstm_9_while_lstm_cell_11_biasadd_readvariableop_resource??
>lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2@
>lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0lstm_9_while_placeholderGlstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype022
0lstm_9/while/TensorArrayV2Read/TensorListGetItem?
/lstm_9/while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp:lstm_9_while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype021
/lstm_9/while/lstm_cell_11/MatMul/ReadVariableOp?
 lstm_9/while/lstm_cell_11/MatMulMatMul7lstm_9/while/TensorArrayV2Read/TensorListGetItem:item:07lstm_9/while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2"
 lstm_9/while/lstm_cell_11/MatMul?
1lstm_9/while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp<lstm_9_while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype023
1lstm_9/while/lstm_cell_11/MatMul_1/ReadVariableOp?
"lstm_9/while/lstm_cell_11/MatMul_1MatMullstm_9_while_placeholder_29lstm_9/while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2$
"lstm_9/while/lstm_cell_11/MatMul_1?
lstm_9/while/lstm_cell_11/addAddV2*lstm_9/while/lstm_cell_11/MatMul:product:0,lstm_9/while/lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_9/while/lstm_cell_11/add?
0lstm_9/while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp;lstm_9_while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype022
0lstm_9/while/lstm_cell_11/BiasAdd/ReadVariableOp?
!lstm_9/while/lstm_cell_11/BiasAddBiasAdd!lstm_9/while/lstm_cell_11/add:z:08lstm_9/while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2#
!lstm_9/while/lstm_cell_11/BiasAdd?
lstm_9/while/lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
lstm_9/while/lstm_cell_11/Const?
)lstm_9/while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)lstm_9/while/lstm_cell_11/split/split_dim?
lstm_9/while/lstm_cell_11/splitSplit2lstm_9/while/lstm_cell_11/split/split_dim:output:0*lstm_9/while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2!
lstm_9/while/lstm_cell_11/split?
!lstm_9/while/lstm_cell_11/SigmoidSigmoid(lstm_9/while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????2#
!lstm_9/while/lstm_cell_11/Sigmoid?
#lstm_9/while/lstm_cell_11/Sigmoid_1Sigmoid(lstm_9/while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????2%
#lstm_9/while/lstm_cell_11/Sigmoid_1?
lstm_9/while/lstm_cell_11/mulMul'lstm_9/while/lstm_cell_11/Sigmoid_1:y:0lstm_9_while_placeholder_3*
T0*'
_output_shapes
:?????????2
lstm_9/while/lstm_cell_11/mul?
lstm_9/while/lstm_cell_11/ReluRelu(lstm_9/while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2 
lstm_9/while/lstm_cell_11/Relu?
lstm_9/while/lstm_cell_11/mul_1Mul%lstm_9/while/lstm_cell_11/Sigmoid:y:0,lstm_9/while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2!
lstm_9/while/lstm_cell_11/mul_1?
lstm_9/while/lstm_cell_11/add_1AddV2!lstm_9/while/lstm_cell_11/mul:z:0#lstm_9/while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2!
lstm_9/while/lstm_cell_11/add_1?
#lstm_9/while/lstm_cell_11/Sigmoid_2Sigmoid(lstm_9/while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????2%
#lstm_9/while/lstm_cell_11/Sigmoid_2?
 lstm_9/while/lstm_cell_11/Relu_1Relu#lstm_9/while/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2"
 lstm_9/while/lstm_cell_11/Relu_1?
lstm_9/while/lstm_cell_11/mul_2Mul'lstm_9/while/lstm_cell_11/Sigmoid_2:y:0.lstm_9/while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2!
lstm_9/while/lstm_cell_11/mul_2?
1lstm_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_9_while_placeholder_1lstm_9_while_placeholder#lstm_9/while/lstm_cell_11/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_9/while/TensorArrayV2Write/TensorListSetItemj
lstm_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/while/add/y?
lstm_9/while/addAddV2lstm_9_while_placeholderlstm_9/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_9/while/addn
lstm_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/while/add_1/y?
lstm_9/while/add_1AddV2&lstm_9_while_lstm_9_while_loop_counterlstm_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_9/while/add_1s
lstm_9/while/IdentityIdentitylstm_9/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_9/while/Identity?
lstm_9/while/Identity_1Identity,lstm_9_while_lstm_9_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_9/while/Identity_1u
lstm_9/while/Identity_2Identitylstm_9/while/add:z:0*
T0*
_output_shapes
: 2
lstm_9/while/Identity_2?
lstm_9/while/Identity_3IdentityAlstm_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_9/while/Identity_3?
lstm_9/while/Identity_4Identity#lstm_9/while/lstm_cell_11/mul_2:z:0*
T0*'
_output_shapes
:?????????2
lstm_9/while/Identity_4?
lstm_9/while/Identity_5Identity#lstm_9/while/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_9/while/Identity_5"7
lstm_9_while_identitylstm_9/while/Identity:output:0";
lstm_9_while_identity_1 lstm_9/while/Identity_1:output:0";
lstm_9_while_identity_2 lstm_9/while/Identity_2:output:0";
lstm_9_while_identity_3 lstm_9/while/Identity_3:output:0";
lstm_9_while_identity_4 lstm_9/while/Identity_4:output:0";
lstm_9_while_identity_5 lstm_9/while/Identity_5:output:0"L
#lstm_9_while_lstm_9_strided_slice_1%lstm_9_while_lstm_9_strided_slice_1_0"x
9lstm_9_while_lstm_cell_11_biasadd_readvariableop_resource;lstm_9_while_lstm_cell_11_biasadd_readvariableop_resource_0"z
:lstm_9_while_lstm_cell_11_matmul_1_readvariableop_resource<lstm_9_while_lstm_cell_11_matmul_1_readvariableop_resource_0"v
8lstm_9_while_lstm_cell_11_matmul_readvariableop_resource:lstm_9_while_lstm_cell_11_matmul_readvariableop_resource_0"?
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensoralstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0*Q
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
?
?
'__inference_lstm_9_layer_call_fn_240979

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
B__inference_lstm_9_layer_call_and_return_conditional_losses_2382972
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
?
L
0__inference_max_pooling1d_1_layer_call_fn_236188

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
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_2361822
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
?
?
'__inference_lstm_9_layer_call_fn_240968

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
B__inference_lstm_9_layer_call_and_return_conditional_losses_2381442
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
?
?
while_cond_239734
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_239734___redundant_placeholder04
0while_while_cond_239734___redundant_placeholder14
0while_while_cond_239734___redundant_placeholder24
0while_while_cond_239734___redundant_placeholder3
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
while_cond_240390
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_240390___redundant_placeholder04
0while_while_cond_240390___redundant_placeholder14
0while_while_cond_240390___redundant_placeholder24
0while_while_cond_240390___redundant_placeholder3
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
?
F
*__inference_flatten_1_layer_call_fn_241059

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
E__inference_flatten_1_layer_call_and_return_conditional_losses_2362852
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
?
?
'__inference_lstm_8_layer_call_fn_239995
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
B__inference_lstm_8_layer_call_and_return_conditional_losses_2369582
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
?
?
L__inference_time_distributed_layer_call_and_return_conditional_losses_236035

inputs
conv1d_2_236024
conv1d_2_236026
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
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv1d_2_236024conv1d_2_236026*
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
D__inference_conv1d_2_layer_call_and_return_conditional_losses_2359382"
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
?	
?
lstm_8_while_cond_238733*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3,
(lstm_8_while_less_lstm_8_strided_slice_1B
>lstm_8_while_lstm_8_while_cond_238733___redundant_placeholder0B
>lstm_8_while_lstm_8_while_cond_238733___redundant_placeholder1B
>lstm_8_while_lstm_8_while_cond_238733___redundant_placeholder2B
>lstm_8_while_lstm_8_while_cond_238733___redundant_placeholder3
lstm_8_while_identity
?
lstm_8/while/LessLesslstm_8_while_placeholder(lstm_8_while_less_lstm_8_strided_slice_1*
T0*
_output_shapes
: 2
lstm_8/while/Lessr
lstm_8/while/IdentityIdentitylstm_8/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_8/while/Identity"7
lstm_8_while_identitylstm_8/while/Identity:output:0*S
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
?%
?
while_body_236889
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_10_236913_0
while_lstm_cell_10_236915_0
while_lstm_cell_10_236917_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_10_236913
while_lstm_cell_10_236915
while_lstm_cell_10_236917??*while/lstm_cell_10/StatefulPartitionedCall?
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
*while/lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10_236913_0while_lstm_cell_10_236915_0while_lstm_cell_10_236917_0*
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
GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_2364632,
*while/lstm_cell_10/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_10/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_10/StatefulPartitionedCall:output:1+^while/lstm_cell_10/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_10/StatefulPartitionedCall:output:2+^while/lstm_cell_10/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_10_236913while_lstm_cell_10_236913_0"8
while_lstm_cell_10_236915while_lstm_cell_10_236915_0"8
while_lstm_cell_10_236917while_lstm_cell_10_236917_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2X
*while/lstm_cell_10/StatefulPartitionedCall*while/lstm_cell_10/StatefulPartitionedCall: 
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
3__inference_time_distributed_1_layer_call_fn_239564

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
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_2361362
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
?	
?
lstm_9_while_cond_239275*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3,
(lstm_9_while_less_lstm_9_strided_slice_1B
>lstm_9_while_lstm_9_while_cond_239275___redundant_placeholder0B
>lstm_9_while_lstm_9_while_cond_239275___redundant_placeholder1B
>lstm_9_while_lstm_9_while_cond_239275___redundant_placeholder2B
>lstm_9_while_lstm_9_while_cond_239275___redundant_placeholder3
lstm_9_while_identity
?
lstm_9/while/LessLesslstm_9_while_placeholder(lstm_9_while_less_lstm_9_strided_slice_1*
T0*
_output_shapes
: 2
lstm_9/while/Lessr
lstm_9/while/IdentityIdentitylstm_9/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_9/while/Identity"7
lstm_9_while_identitylstm_9/while/Identity:output:0*S
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
while_cond_237876
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_237876___redundant_placeholder04
0while_while_cond_237876___redundant_placeholder14
0while_while_cond_237876___redundant_placeholder24
0while_while_cond_237876___redundant_placeholder3
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
?
?
D__inference_conv1d_2_layer_call_and_return_conditional_losses_241014

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
?W
?
B__inference_lstm_9_layer_call_and_return_conditional_losses_240629
inputs_0/
+lstm_cell_11_matmul_readvariableop_resource1
-lstm_cell_11_matmul_1_readvariableop_resource0
,lstm_cell_11_biasadd_readvariableop_resource
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
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOp?
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/MatMul?
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOp?
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/MatMul_1?
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/add?
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOp?
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/BiasAddj
lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/Const~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dim?
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_11/split?
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid?
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid_1?
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul}
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Relu?
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul_1?
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/add_1?
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid_2|
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Relu_1?
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
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
while_body_240544*
condR
while_cond_240543*K
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
?9
?
while_body_239888
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_10_matmul_readvariableop_resource_09
5while_lstm_cell_10_matmul_1_readvariableop_resource_08
4while_lstm_cell_10_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_10_matmul_readvariableop_resource7
3while_lstm_cell_10_matmul_1_readvariableop_resource6
2while_lstm_cell_10_biasadd_readvariableop_resource??
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp?
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/MatMul?
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOp?
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/MatMul_1?
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/add?
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOp?
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/BiasAddv
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const?
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim?
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_10/split?
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid?
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid_1?
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul?
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Relu?
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul_1?
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/add_1?
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid_2?
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Relu_1?
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_2:z:0*
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
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
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
B__inference_lstm_9_layer_call_and_return_conditional_losses_240957

inputs/
+lstm_cell_11_matmul_readvariableop_resource1
-lstm_cell_11_matmul_1_readvariableop_resource0
,lstm_cell_11_biasadd_readvariableop_resource
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
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOp?
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/MatMul?
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOp?
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/MatMul_1?
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/add?
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOp?
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_cell_11/BiasAddj
lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/Const~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dim?
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_11/split?
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid?
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid_1?
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul}
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Relu?
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul_1?
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/add_1?
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Sigmoid_2|
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/Relu_1?
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_11/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
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
while_body_240872*
condR
while_cond_240871*K
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
D__inference_conv1d_3_layer_call_and_return_conditional_losses_241039

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
?
?
%sequential_5_lstm_8_while_cond_235670D
@sequential_5_lstm_8_while_sequential_5_lstm_8_while_loop_counterJ
Fsequential_5_lstm_8_while_sequential_5_lstm_8_while_maximum_iterations)
%sequential_5_lstm_8_while_placeholder+
'sequential_5_lstm_8_while_placeholder_1+
'sequential_5_lstm_8_while_placeholder_2+
'sequential_5_lstm_8_while_placeholder_3F
Bsequential_5_lstm_8_while_less_sequential_5_lstm_8_strided_slice_1\
Xsequential_5_lstm_8_while_sequential_5_lstm_8_while_cond_235670___redundant_placeholder0\
Xsequential_5_lstm_8_while_sequential_5_lstm_8_while_cond_235670___redundant_placeholder1\
Xsequential_5_lstm_8_while_sequential_5_lstm_8_while_cond_235670___redundant_placeholder2\
Xsequential_5_lstm_8_while_sequential_5_lstm_8_while_cond_235670___redundant_placeholder3&
"sequential_5_lstm_8_while_identity
?
sequential_5/lstm_8/while/LessLess%sequential_5_lstm_8_while_placeholderBsequential_5_lstm_8_while_less_sequential_5_lstm_8_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_5/lstm_8/while/Less?
"sequential_5/lstm_8/while/IdentityIdentity"sequential_5/lstm_8/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_5/lstm_8/while/Identity"Q
"sequential_5_lstm_8_while_identity+sequential_5/lstm_8/while/Identity:output:0*S
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
?/
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_238515

inputs
time_distributed_238475
time_distributed_238477
time_distributed_1_238482
time_distributed_1_238484
lstm_8_238495
lstm_8_238497
lstm_8_238499
lstm_9_238502
lstm_9_238504
lstm_9_238506
dense_9_238509
dense_9_238511
identity??dense_9/StatefulPartitionedCall?lstm_8/StatefulPartitionedCall?lstm_9/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_238475time_distributed_238477*
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
L__inference_time_distributed_layer_call_and_return_conditional_losses_2360352*
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
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_238482time_distributed_1_238484*
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
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_2361662,
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
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_2362652$
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
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_2363542$
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
lstm_8/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0lstm_8_238495lstm_8_238497lstm_8_238499*
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
B__inference_lstm_8_layer_call_and_return_conditional_losses_2379622 
lstm_8/StatefulPartitionedCall?
lstm_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0lstm_9_238502lstm_9_238504lstm_9_238506*
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
B__inference_lstm_9_layer_call_and_return_conditional_losses_2382972 
lstm_9/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_9/StatefulPartitionedCall:output:0dense_9_238509dense_9_238511*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_2383372!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_9/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:` \
8
_output_shapes&
$:"??????????????????

 
_user_specified_nameinputs
?%
?
while_body_236757
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_10_236781_0
while_lstm_cell_10_236783_0
while_lstm_cell_10_236785_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_10_236781
while_lstm_cell_10_236783
while_lstm_cell_10_236785??*while/lstm_cell_10/StatefulPartitionedCall?
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
*while/lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10_236781_0while_lstm_cell_10_236783_0while_lstm_cell_10_236785_0*
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
GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_2364302,
*while/lstm_cell_10/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_10/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_10/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_10/StatefulPartitionedCall:output:1+^while/lstm_cell_10/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_10/StatefulPartitionedCall:output:2+^while/lstm_cell_10/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_10_236781while_lstm_cell_10_236781_0"8
while_lstm_cell_10_236783while_lstm_cell_10_236783_0"8
while_lstm_cell_10_236785while_lstm_cell_10_236785_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2X
*while/lstm_cell_10/StatefulPartitionedCall*while/lstm_cell_10/StatefulPartitionedCall: 
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
?9
?
while_body_240872
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_11_matmul_readvariableop_resource_09
5while_lstm_cell_11_matmul_1_readvariableop_resource_08
4while_lstm_cell_11_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_11_matmul_readvariableop_resource7
3while_lstm_cell_11_matmul_1_readvariableop_resource6
2while_lstm_cell_11_biasadd_readvariableop_resource??
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
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp?
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/MatMul?
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOp?
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/MatMul_1?
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/add?
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOp?
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/BiasAddv
while/lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_11/Const?
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dim?
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_11/split?
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid?
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid_1?
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul?
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Relu?
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul_1?
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/add_1?
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid_2?
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Relu_1?
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_11/mul_2:z:0*
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
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_11_biasadd_readvariableop_resource4while_lstm_cell_11_biasadd_readvariableop_resource_0"l
3while_lstm_cell_11_matmul_1_readvariableop_resource5while_lstm_cell_11_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_11_matmul_readvariableop_resource3while_lstm_cell_11_matmul_readvariableop_resource_0"0
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
?
%sequential_5_lstm_9_while_cond_235819D
@sequential_5_lstm_9_while_sequential_5_lstm_9_while_loop_counterJ
Fsequential_5_lstm_9_while_sequential_5_lstm_9_while_maximum_iterations)
%sequential_5_lstm_9_while_placeholder+
'sequential_5_lstm_9_while_placeholder_1+
'sequential_5_lstm_9_while_placeholder_2+
'sequential_5_lstm_9_while_placeholder_3F
Bsequential_5_lstm_9_while_less_sequential_5_lstm_9_strided_slice_1\
Xsequential_5_lstm_9_while_sequential_5_lstm_9_while_cond_235819___redundant_placeholder0\
Xsequential_5_lstm_9_while_sequential_5_lstm_9_while_cond_235819___redundant_placeholder1\
Xsequential_5_lstm_9_while_sequential_5_lstm_9_while_cond_235819___redundant_placeholder2\
Xsequential_5_lstm_9_while_sequential_5_lstm_9_while_cond_235819___redundant_placeholder3&
"sequential_5_lstm_9_while_identity
?
sequential_5/lstm_9/while/LessLess%sequential_5_lstm_9_while_placeholderBsequential_5_lstm_9_while_less_sequential_5_lstm_9_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_5/lstm_9/while/Less?
"sequential_5/lstm_9/while/IdentityIdentity"sequential_5/lstm_9/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_5/lstm_9/while/Identity"Q
"sequential_5_lstm_9_while_identity+sequential_5/lstm_9/while/Identity:output:0*S
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
3__inference_time_distributed_1_layer_call_fn_239573

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
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_2361662
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
?/
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_238443

inputs
time_distributed_238403
time_distributed_238405
time_distributed_1_238410
time_distributed_1_238412
lstm_8_238423
lstm_8_238425
lstm_8_238427
lstm_9_238430
lstm_9_238432
lstm_9_238434
dense_9_238437
dense_9_238439
identity??dense_9/StatefulPartitionedCall?lstm_8/StatefulPartitionedCall?lstm_9/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_238403time_distributed_238405*
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
L__inference_time_distributed_layer_call_and_return_conditional_losses_2360052*
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
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_238410time_distributed_1_238412*
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
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_2361362,
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
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_2362432$
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
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_2363332$
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
lstm_8/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0lstm_8_238423lstm_8_238425lstm_8_238427*
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
B__inference_lstm_8_layer_call_and_return_conditional_losses_2378092 
lstm_8/StatefulPartitionedCall?
lstm_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0lstm_9_238430lstm_9_238432lstm_9_238434*
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
B__inference_lstm_9_layer_call_and_return_conditional_losses_2381442 
lstm_9/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_9/StatefulPartitionedCall:output:0dense_9_238437dense_9_238439*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_2383372!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_9/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:` \
8
_output_shapes&
$:"??????????????????

 
_user_specified_nameinputs
?9
?
while_body_238059
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_11_matmul_readvariableop_resource_09
5while_lstm_cell_11_matmul_1_readvariableop_resource_08
4while_lstm_cell_11_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_11_matmul_readvariableop_resource7
3while_lstm_cell_11_matmul_1_readvariableop_resource6
2while_lstm_cell_11_biasadd_readvariableop_resource??
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
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp?
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/MatMul?
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOp?
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/MatMul_1?
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/add?
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOp?
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/BiasAddv
while/lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_11/Const?
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dim?
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_11/split?
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid?
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid_1?
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul?
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Relu?
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul_1?
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/add_1?
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid_2?
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Relu_1?
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_11/mul_2:z:0*
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
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_11_biasadd_readvariableop_resource4while_lstm_cell_11_biasadd_readvariableop_resource_0"l
3while_lstm_cell_11_matmul_1_readvariableop_resource5while_lstm_cell_11_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_11_matmul_readvariableop_resource3while_lstm_cell_11_matmul_readvariableop_resource_0"0
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
?
lstm_8_while_cond_239126*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3,
(lstm_8_while_less_lstm_8_strided_slice_1B
>lstm_8_while_lstm_8_while_cond_239126___redundant_placeholder0B
>lstm_8_while_lstm_8_while_cond_239126___redundant_placeholder1B
>lstm_8_while_lstm_8_while_cond_239126___redundant_placeholder2B
>lstm_8_while_lstm_8_while_cond_239126___redundant_placeholder3
lstm_8_while_identity
?
lstm_8/while/LessLesslstm_8_while_placeholder(lstm_8_while_less_lstm_8_strided_slice_1*
T0*
_output_shapes
: 2
lstm_8/while/Lessr
lstm_8/while/IdentityIdentitylstm_8/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_8/while/Identity"7
lstm_8_while_identitylstm_8/while/Identity:output:0*S
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
?
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_236265

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
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_2361822!
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
?
?
C__inference_dense_9_layer_call_and_return_conditional_losses_240989

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_238974

inputsI
Etime_distributed_conv1d_2_conv1d_expanddims_1_readvariableop_resource=
9time_distributed_conv1d_2_biasadd_readvariableop_resourceK
Gtime_distributed_1_conv1d_3_conv1d_expanddims_1_readvariableop_resource?
;time_distributed_1_conv1d_3_biasadd_readvariableop_resource6
2lstm_8_lstm_cell_10_matmul_readvariableop_resource8
4lstm_8_lstm_cell_10_matmul_1_readvariableop_resource7
3lstm_8_lstm_cell_10_biasadd_readvariableop_resource6
2lstm_9_lstm_cell_11_matmul_readvariableop_resource8
4lstm_9_lstm_cell_11_matmul_1_readvariableop_resource7
3lstm_9_lstm_cell_11_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identity??lstm_8/while?lstm_9/whilef
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
lstm_8/ShapeShape%time_distributed_3/Reshape_1:output:0*
T0*
_output_shapes
:2
lstm_8/Shape?
lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice/stack?
lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_1?
lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_2?
lstm_8/strided_sliceStridedSlicelstm_8/Shape:output:0#lstm_8/strided_slice/stack:output:0%lstm_8/strided_slice/stack_1:output:0%lstm_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slicej
lstm_8/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_8/zeros/mul/y?
lstm_8/zeros/mulMullstm_8/strided_slice:output:0lstm_8/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/mulm
lstm_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_8/zeros/Less/y?
lstm_8/zeros/LessLesslstm_8/zeros/mul:z:0lstm_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/Lessp
lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_8/zeros/packed/1?
lstm_8/zeros/packedPacklstm_8/strided_slice:output:0lstm_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_8/zeros/packedm
lstm_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/zeros/Const?
lstm_8/zerosFilllstm_8/zeros/packed:output:0lstm_8/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_8/zerosn
lstm_8/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_8/zeros_1/mul/y?
lstm_8/zeros_1/mulMullstm_8/strided_slice:output:0lstm_8/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/mulq
lstm_8/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_8/zeros_1/Less/y?
lstm_8/zeros_1/LessLesslstm_8/zeros_1/mul:z:0lstm_8/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/Lesst
lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_8/zeros_1/packed/1?
lstm_8/zeros_1/packedPacklstm_8/strided_slice:output:0 lstm_8/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_8/zeros_1/packedq
lstm_8/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/zeros_1/Const?
lstm_8/zeros_1Filllstm_8/zeros_1/packed:output:0lstm_8/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_8/zeros_1?
lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose/perm?
lstm_8/transpose	Transpose%time_distributed_3/Reshape_1:output:0lstm_8/transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
lstm_8/transposed
lstm_8/Shape_1Shapelstm_8/transpose:y:0*
T0*
_output_shapes
:2
lstm_8/Shape_1?
lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_1/stack?
lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_1?
lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_2?
lstm_8/strided_slice_1StridedSlicelstm_8/Shape_1:output:0%lstm_8/strided_slice_1/stack:output:0'lstm_8/strided_slice_1/stack_1:output:0'lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slice_1?
"lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_8/TensorArrayV2/element_shape?
lstm_8/TensorArrayV2TensorListReserve+lstm_8/TensorArrayV2/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2?
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2>
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_8/transpose:y:0Elstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_8/TensorArrayUnstack/TensorListFromTensor?
lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_2/stack?
lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_1?
lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_2?
lstm_8/strided_slice_2StridedSlicelstm_8/transpose:y:0%lstm_8/strided_slice_2/stack:output:0'lstm_8/strided_slice_2/stack_1:output:0'lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_8/strided_slice_2?
)lstm_8/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp2lstm_8_lstm_cell_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)lstm_8/lstm_cell_10/MatMul/ReadVariableOp?
lstm_8/lstm_cell_10/MatMulMatMullstm_8/strided_slice_2:output:01lstm_8/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_10/MatMul?
+lstm_8/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp4lstm_8_lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02-
+lstm_8/lstm_cell_10/MatMul_1/ReadVariableOp?
lstm_8/lstm_cell_10/MatMul_1MatMullstm_8/zeros:output:03lstm_8/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_10/MatMul_1?
lstm_8/lstm_cell_10/addAddV2$lstm_8/lstm_cell_10/MatMul:product:0&lstm_8/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_10/add?
*lstm_8/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp3lstm_8_lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*lstm_8/lstm_cell_10/BiasAdd/ReadVariableOp?
lstm_8/lstm_cell_10/BiasAddBiasAddlstm_8/lstm_cell_10/add:z:02lstm_8/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_8/lstm_cell_10/BiasAddx
lstm_8/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/lstm_cell_10/Const?
#lstm_8/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#lstm_8/lstm_cell_10/split/split_dim?
lstm_8/lstm_cell_10/splitSplit,lstm_8/lstm_cell_10/split/split_dim:output:0$lstm_8/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_8/lstm_cell_10/split?
lstm_8/lstm_cell_10/SigmoidSigmoid"lstm_8/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/Sigmoid?
lstm_8/lstm_cell_10/Sigmoid_1Sigmoid"lstm_8/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/Sigmoid_1?
lstm_8/lstm_cell_10/mulMul!lstm_8/lstm_cell_10/Sigmoid_1:y:0lstm_8/zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/mul?
lstm_8/lstm_cell_10/ReluRelu"lstm_8/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/Relu?
lstm_8/lstm_cell_10/mul_1Mullstm_8/lstm_cell_10/Sigmoid:y:0&lstm_8/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/mul_1?
lstm_8/lstm_cell_10/add_1AddV2lstm_8/lstm_cell_10/mul:z:0lstm_8/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/add_1?
lstm_8/lstm_cell_10/Sigmoid_2Sigmoid"lstm_8/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/Sigmoid_2?
lstm_8/lstm_cell_10/Relu_1Relulstm_8/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/Relu_1?
lstm_8/lstm_cell_10/mul_2Mul!lstm_8/lstm_cell_10/Sigmoid_2:y:0(lstm_8/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_8/lstm_cell_10/mul_2?
$lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2&
$lstm_8/TensorArrayV2_1/element_shape?
lstm_8/TensorArrayV2_1TensorListReserve-lstm_8/TensorArrayV2_1/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2_1\
lstm_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/time?
lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_8/while/maximum_iterationsx
lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/while/loop_counter?
lstm_8/whileWhile"lstm_8/while/loop_counter:output:0(lstm_8/while/maximum_iterations:output:0lstm_8/time:output:0lstm_8/TensorArrayV2_1:handle:0lstm_8/zeros:output:0lstm_8/zeros_1:output:0lstm_8/strided_slice_1:output:0>lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_8_lstm_cell_10_matmul_readvariableop_resource4lstm_8_lstm_cell_10_matmul_1_readvariableop_resource3lstm_8_lstm_cell_10_biasadd_readvariableop_resource*
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
lstm_8_while_body_238734*$
condR
lstm_8_while_cond_238733*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
lstm_8/while?
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_8/TensorArrayV2Stack/TensorListStackTensorListStacklstm_8/while:output:3@lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02+
)lstm_8/TensorArrayV2Stack/TensorListStack?
lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_8/strided_slice_3/stack?
lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_8/strided_slice_3/stack_1?
lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_3/stack_2?
lstm_8/strided_slice_3StridedSlice2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_8/strided_slice_3/stack:output:0'lstm_8/strided_slice_3/stack_1:output:0'lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_8/strided_slice_3?
lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose_1/perm?
lstm_8/transpose_1	Transpose2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_8/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
lstm_8/transpose_1t
lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/runtimeb
lstm_9/ShapeShapelstm_8/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_9/Shape?
lstm_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice/stack?
lstm_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_9/strided_slice/stack_1?
lstm_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_9/strided_slice/stack_2?
lstm_9/strided_sliceStridedSlicelstm_9/Shape:output:0#lstm_9/strided_slice/stack:output:0%lstm_9/strided_slice/stack_1:output:0%lstm_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_9/strided_slicej
lstm_9/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/zeros/mul/y?
lstm_9/zeros/mulMullstm_9/strided_slice:output:0lstm_9/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros/mulm
lstm_9/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_9/zeros/Less/y?
lstm_9/zeros/LessLesslstm_9/zeros/mul:z:0lstm_9/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros/Lessp
lstm_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/zeros/packed/1?
lstm_9/zeros/packedPacklstm_9/strided_slice:output:0lstm_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_9/zeros/packedm
lstm_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/zeros/Const?
lstm_9/zerosFilllstm_9/zeros/packed:output:0lstm_9/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_9/zerosn
lstm_9/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/zeros_1/mul/y?
lstm_9/zeros_1/mulMullstm_9/strided_slice:output:0lstm_9/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros_1/mulq
lstm_9/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_9/zeros_1/Less/y?
lstm_9/zeros_1/LessLesslstm_9/zeros_1/mul:z:0lstm_9/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros_1/Lesst
lstm_9/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/zeros_1/packed/1?
lstm_9/zeros_1/packedPacklstm_9/strided_slice:output:0 lstm_9/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_9/zeros_1/packedq
lstm_9/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/zeros_1/Const?
lstm_9/zeros_1Filllstm_9/zeros_1/packed:output:0lstm_9/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lstm_9/zeros_1?
lstm_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_9/transpose/perm?
lstm_9/transpose	Transposelstm_8/transpose_1:y:0lstm_9/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
lstm_9/transposed
lstm_9/Shape_1Shapelstm_9/transpose:y:0*
T0*
_output_shapes
:2
lstm_9/Shape_1?
lstm_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice_1/stack?
lstm_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_1/stack_1?
lstm_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_1/stack_2?
lstm_9/strided_slice_1StridedSlicelstm_9/Shape_1:output:0%lstm_9/strided_slice_1/stack:output:0'lstm_9/strided_slice_1/stack_1:output:0'lstm_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_9/strided_slice_1?
"lstm_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_9/TensorArrayV2/element_shape?
lstm_9/TensorArrayV2TensorListReserve+lstm_9/TensorArrayV2/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_9/TensorArrayV2?
<lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2>
<lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_9/transpose:y:0Elstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_9/TensorArrayUnstack/TensorListFromTensor?
lstm_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice_2/stack?
lstm_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_2/stack_1?
lstm_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_2/stack_2?
lstm_9/strided_slice_2StridedSlicelstm_9/transpose:y:0%lstm_9/strided_slice_2/stack:output:0'lstm_9/strided_slice_2/stack_1:output:0'lstm_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_9/strided_slice_2?
)lstm_9/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp2lstm_9_lstm_cell_11_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02+
)lstm_9/lstm_cell_11/MatMul/ReadVariableOp?
lstm_9/lstm_cell_11/MatMulMatMullstm_9/strided_slice_2:output:01lstm_9/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_9/lstm_cell_11/MatMul?
+lstm_9/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp4lstm_9_lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02-
+lstm_9/lstm_cell_11/MatMul_1/ReadVariableOp?
lstm_9/lstm_cell_11/MatMul_1MatMullstm_9/zeros:output:03lstm_9/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_9/lstm_cell_11/MatMul_1?
lstm_9/lstm_cell_11/addAddV2$lstm_9/lstm_cell_11/MatMul:product:0&lstm_9/lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
lstm_9/lstm_cell_11/add?
*lstm_9/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp3lstm_9_lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02,
*lstm_9/lstm_cell_11/BiasAdd/ReadVariableOp?
lstm_9/lstm_cell_11/BiasAddBiasAddlstm_9/lstm_cell_11/add:z:02lstm_9/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
lstm_9/lstm_cell_11/BiasAddx
lstm_9/lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/lstm_cell_11/Const?
#lstm_9/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#lstm_9/lstm_cell_11/split/split_dim?
lstm_9/lstm_cell_11/splitSplit,lstm_9/lstm_cell_11/split/split_dim:output:0$lstm_9/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_9/lstm_cell_11/split?
lstm_9/lstm_cell_11/SigmoidSigmoid"lstm_9/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/Sigmoid?
lstm_9/lstm_cell_11/Sigmoid_1Sigmoid"lstm_9/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/Sigmoid_1?
lstm_9/lstm_cell_11/mulMul!lstm_9/lstm_cell_11/Sigmoid_1:y:0lstm_9/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/mul?
lstm_9/lstm_cell_11/ReluRelu"lstm_9/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/Relu?
lstm_9/lstm_cell_11/mul_1Mullstm_9/lstm_cell_11/Sigmoid:y:0&lstm_9/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/mul_1?
lstm_9/lstm_cell_11/add_1AddV2lstm_9/lstm_cell_11/mul:z:0lstm_9/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/add_1?
lstm_9/lstm_cell_11/Sigmoid_2Sigmoid"lstm_9/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/Sigmoid_2?
lstm_9/lstm_cell_11/Relu_1Relulstm_9/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/Relu_1?
lstm_9/lstm_cell_11/mul_2Mul!lstm_9/lstm_cell_11/Sigmoid_2:y:0(lstm_9/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_9/lstm_cell_11/mul_2?
$lstm_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$lstm_9/TensorArrayV2_1/element_shape?
lstm_9/TensorArrayV2_1TensorListReserve-lstm_9/TensorArrayV2_1/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_9/TensorArrayV2_1\
lstm_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_9/time?
lstm_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_9/while/maximum_iterationsx
lstm_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_9/while/loop_counter?
lstm_9/whileWhile"lstm_9/while/loop_counter:output:0(lstm_9/while/maximum_iterations:output:0lstm_9/time:output:0lstm_9/TensorArrayV2_1:handle:0lstm_9/zeros:output:0lstm_9/zeros_1:output:0lstm_9/strided_slice_1:output:0>lstm_9/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_9_lstm_cell_11_matmul_readvariableop_resource4lstm_9_lstm_cell_11_matmul_1_readvariableop_resource3lstm_9_lstm_cell_11_biasadd_readvariableop_resource*
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
lstm_9_while_body_238883*$
condR
lstm_9_while_cond_238882*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
lstm_9/while?
7lstm_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7lstm_9/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_9/TensorArrayV2Stack/TensorListStackTensorListStacklstm_9/while:output:3@lstm_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02+
)lstm_9/TensorArrayV2Stack/TensorListStack?
lstm_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_9/strided_slice_3/stack?
lstm_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_9/strided_slice_3/stack_1?
lstm_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_3/stack_2?
lstm_9/strided_slice_3StridedSlice2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_9/strided_slice_3/stack:output:0'lstm_9/strided_slice_3/stack_1:output:0'lstm_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_9/strided_slice_3?
lstm_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_9/transpose_1/perm?
lstm_9/transpose_1	Transpose2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_9/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
lstm_9/transpose_1t
lstm_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/runtime?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMullstm_9/strided_slice_3:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense_9/BiasAdd?
IdentityIdentitydense_9/BiasAdd:output:0^lstm_8/while^lstm_9/while*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::2
lstm_8/whilelstm_8/while2
lstm_9/whilelstm_9/while:` \
8
_output_shapes&
$:"??????????????????

 
_user_specified_nameinputs
?
?
while_cond_238058
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_238058___redundant_placeholder04
0while_while_cond_238058___redundant_placeholder14
0while_while_cond_238058___redundant_placeholder24
0while_while_cond_238058___redundant_placeholder3
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
L__inference_time_distributed_layer_call_and_return_conditional_losses_236005

inputs
conv1d_2_235994
conv1d_2_235996
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
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv1d_2_235994conv1d_2_235996*
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
D__inference_conv1d_2_layer_call_and_return_conditional_losses_2359382"
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
?%
?
while_body_237499
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_11_237523_0
while_lstm_cell_11_237525_0
while_lstm_cell_11_237527_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_11_237523
while_lstm_cell_11_237525
while_lstm_cell_11_237527??*while/lstm_cell_11/StatefulPartitionedCall?
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
*while/lstm_cell_11/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_11_237523_0while_lstm_cell_11_237525_0while_lstm_cell_11_237527_0*
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
GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_11_layer_call_and_return_conditional_losses_2370732,
*while/lstm_cell_11/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_11/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_11/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_11/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_11/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_11/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_11/StatefulPartitionedCall:output:1+^while/lstm_cell_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_11/StatefulPartitionedCall:output:2+^while/lstm_cell_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_11_237523while_lstm_cell_11_237523_0"8
while_lstm_cell_11_237525while_lstm_cell_11_237525_0"8
while_lstm_cell_11_237527while_lstm_cell_11_237527_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2X
*while/lstm_cell_11/StatefulPartitionedCall*while/lstm_cell_11/StatefulPartitionedCall: 
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
)__inference_conv1d_2_layer_call_fn_241023

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
D__inference_conv1d_2_layer_call_and_return_conditional_losses_2359382
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
?B
?
lstm_8_while_body_238734*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3)
%lstm_8_while_lstm_8_strided_slice_1_0e
alstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0>
:lstm_8_while_lstm_cell_10_matmul_readvariableop_resource_0@
<lstm_8_while_lstm_cell_10_matmul_1_readvariableop_resource_0?
;lstm_8_while_lstm_cell_10_biasadd_readvariableop_resource_0
lstm_8_while_identity
lstm_8_while_identity_1
lstm_8_while_identity_2
lstm_8_while_identity_3
lstm_8_while_identity_4
lstm_8_while_identity_5'
#lstm_8_while_lstm_8_strided_slice_1c
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor<
8lstm_8_while_lstm_cell_10_matmul_readvariableop_resource>
:lstm_8_while_lstm_cell_10_matmul_1_readvariableop_resource=
9lstm_8_while_lstm_cell_10_biasadd_readvariableop_resource??
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2@
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0lstm_8_while_placeholderGlstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype022
0lstm_8/while/TensorArrayV2Read/TensorListGetItem?
/lstm_8/while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp:lstm_8_while_lstm_cell_10_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/lstm_8/while/lstm_cell_10/MatMul/ReadVariableOp?
 lstm_8/while/lstm_cell_10/MatMulMatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:07lstm_8/while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 lstm_8/while/lstm_cell_10/MatMul?
1lstm_8/while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp<lstm_8_while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype023
1lstm_8/while/lstm_cell_10/MatMul_1/ReadVariableOp?
"lstm_8/while/lstm_cell_10/MatMul_1MatMullstm_8_while_placeholder_29lstm_8/while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_8/while/lstm_cell_10/MatMul_1?
lstm_8/while/lstm_cell_10/addAddV2*lstm_8/while/lstm_cell_10/MatMul:product:0,lstm_8/while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_8/while/lstm_cell_10/add?
0lstm_8/while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp;lstm_8_while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype022
0lstm_8/while/lstm_cell_10/BiasAdd/ReadVariableOp?
!lstm_8/while/lstm_cell_10/BiasAddBiasAdd!lstm_8/while/lstm_cell_10/add:z:08lstm_8/while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_8/while/lstm_cell_10/BiasAdd?
lstm_8/while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
lstm_8/while/lstm_cell_10/Const?
)lstm_8/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)lstm_8/while/lstm_cell_10/split/split_dim?
lstm_8/while/lstm_cell_10/splitSplit2lstm_8/while/lstm_cell_10/split/split_dim:output:0*lstm_8/while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2!
lstm_8/while/lstm_cell_10/split?
!lstm_8/while/lstm_cell_10/SigmoidSigmoid(lstm_8/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????22#
!lstm_8/while/lstm_cell_10/Sigmoid?
#lstm_8/while/lstm_cell_10/Sigmoid_1Sigmoid(lstm_8/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????22%
#lstm_8/while/lstm_cell_10/Sigmoid_1?
lstm_8/while/lstm_cell_10/mulMul'lstm_8/while/lstm_cell_10/Sigmoid_1:y:0lstm_8_while_placeholder_3*
T0*'
_output_shapes
:?????????22
lstm_8/while/lstm_cell_10/mul?
lstm_8/while/lstm_cell_10/ReluRelu(lstm_8/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22 
lstm_8/while/lstm_cell_10/Relu?
lstm_8/while/lstm_cell_10/mul_1Mul%lstm_8/while/lstm_cell_10/Sigmoid:y:0,lstm_8/while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22!
lstm_8/while/lstm_cell_10/mul_1?
lstm_8/while/lstm_cell_10/add_1AddV2!lstm_8/while/lstm_cell_10/mul:z:0#lstm_8/while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22!
lstm_8/while/lstm_cell_10/add_1?
#lstm_8/while/lstm_cell_10/Sigmoid_2Sigmoid(lstm_8/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????22%
#lstm_8/while/lstm_cell_10/Sigmoid_2?
 lstm_8/while/lstm_cell_10/Relu_1Relu#lstm_8/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22"
 lstm_8/while/lstm_cell_10/Relu_1?
lstm_8/while/lstm_cell_10/mul_2Mul'lstm_8/while/lstm_cell_10/Sigmoid_2:y:0.lstm_8/while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22!
lstm_8/while/lstm_cell_10/mul_2?
1lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_8_while_placeholder_1lstm_8_while_placeholder#lstm_8/while/lstm_cell_10/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_8/while/TensorArrayV2Write/TensorListSetItemj
lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add/y?
lstm_8/while/addAddV2lstm_8_while_placeholderlstm_8/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/addn
lstm_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add_1/y?
lstm_8/while/add_1AddV2&lstm_8_while_lstm_8_while_loop_counterlstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/add_1s
lstm_8/while/IdentityIdentitylstm_8/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_8/while/Identity?
lstm_8/while/Identity_1Identity,lstm_8_while_lstm_8_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_8/while/Identity_1u
lstm_8/while/Identity_2Identitylstm_8/while/add:z:0*
T0*
_output_shapes
: 2
lstm_8/while/Identity_2?
lstm_8/while/Identity_3IdentityAlstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_8/while/Identity_3?
lstm_8/while/Identity_4Identity#lstm_8/while/lstm_cell_10/mul_2:z:0*
T0*'
_output_shapes
:?????????22
lstm_8/while/Identity_4?
lstm_8/while/Identity_5Identity#lstm_8/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_8/while/Identity_5"7
lstm_8_while_identitylstm_8/while/Identity:output:0";
lstm_8_while_identity_1 lstm_8/while/Identity_1:output:0";
lstm_8_while_identity_2 lstm_8/while/Identity_2:output:0";
lstm_8_while_identity_3 lstm_8/while/Identity_3:output:0";
lstm_8_while_identity_4 lstm_8/while/Identity_4:output:0";
lstm_8_while_identity_5 lstm_8/while/Identity_5:output:0"L
#lstm_8_while_lstm_8_strided_slice_1%lstm_8_while_lstm_8_strided_slice_1_0"x
9lstm_8_while_lstm_cell_10_biasadd_readvariableop_resource;lstm_8_while_lstm_cell_10_biasadd_readvariableop_resource_0"z
:lstm_8_while_lstm_cell_10_matmul_1_readvariableop_resource<lstm_8_while_lstm_cell_10_matmul_1_readvariableop_resource_0"v
8lstm_8_while_lstm_cell_10_matmul_readvariableop_resource:lstm_8_while_lstm_cell_10_matmul_readvariableop_resource_0"?
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensoralstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*Q
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
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_236166

inputs
conv1d_3_236155
conv1d_3_236157
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
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv1d_3_236155conv1d_3_236157*
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
D__inference_conv1d_3_layer_call_and_return_conditional_losses_2360692"
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
?B
?
lstm_8_while_body_239127*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3)
%lstm_8_while_lstm_8_strided_slice_1_0e
alstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0>
:lstm_8_while_lstm_cell_10_matmul_readvariableop_resource_0@
<lstm_8_while_lstm_cell_10_matmul_1_readvariableop_resource_0?
;lstm_8_while_lstm_cell_10_biasadd_readvariableop_resource_0
lstm_8_while_identity
lstm_8_while_identity_1
lstm_8_while_identity_2
lstm_8_while_identity_3
lstm_8_while_identity_4
lstm_8_while_identity_5'
#lstm_8_while_lstm_8_strided_slice_1c
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor<
8lstm_8_while_lstm_cell_10_matmul_readvariableop_resource>
:lstm_8_while_lstm_cell_10_matmul_1_readvariableop_resource=
9lstm_8_while_lstm_cell_10_biasadd_readvariableop_resource??
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2@
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0lstm_8_while_placeholderGlstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype022
0lstm_8/while/TensorArrayV2Read/TensorListGetItem?
/lstm_8/while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp:lstm_8_while_lstm_cell_10_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/lstm_8/while/lstm_cell_10/MatMul/ReadVariableOp?
 lstm_8/while/lstm_cell_10/MatMulMatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:07lstm_8/while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 lstm_8/while/lstm_cell_10/MatMul?
1lstm_8/while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp<lstm_8_while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype023
1lstm_8/while/lstm_cell_10/MatMul_1/ReadVariableOp?
"lstm_8/while/lstm_cell_10/MatMul_1MatMullstm_8_while_placeholder_29lstm_8/while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_8/while/lstm_cell_10/MatMul_1?
lstm_8/while/lstm_cell_10/addAddV2*lstm_8/while/lstm_cell_10/MatMul:product:0,lstm_8/while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_8/while/lstm_cell_10/add?
0lstm_8/while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp;lstm_8_while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype022
0lstm_8/while/lstm_cell_10/BiasAdd/ReadVariableOp?
!lstm_8/while/lstm_cell_10/BiasAddBiasAdd!lstm_8/while/lstm_cell_10/add:z:08lstm_8/while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_8/while/lstm_cell_10/BiasAdd?
lstm_8/while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
lstm_8/while/lstm_cell_10/Const?
)lstm_8/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)lstm_8/while/lstm_cell_10/split/split_dim?
lstm_8/while/lstm_cell_10/splitSplit2lstm_8/while/lstm_cell_10/split/split_dim:output:0*lstm_8/while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2!
lstm_8/while/lstm_cell_10/split?
!lstm_8/while/lstm_cell_10/SigmoidSigmoid(lstm_8/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????22#
!lstm_8/while/lstm_cell_10/Sigmoid?
#lstm_8/while/lstm_cell_10/Sigmoid_1Sigmoid(lstm_8/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????22%
#lstm_8/while/lstm_cell_10/Sigmoid_1?
lstm_8/while/lstm_cell_10/mulMul'lstm_8/while/lstm_cell_10/Sigmoid_1:y:0lstm_8_while_placeholder_3*
T0*'
_output_shapes
:?????????22
lstm_8/while/lstm_cell_10/mul?
lstm_8/while/lstm_cell_10/ReluRelu(lstm_8/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22 
lstm_8/while/lstm_cell_10/Relu?
lstm_8/while/lstm_cell_10/mul_1Mul%lstm_8/while/lstm_cell_10/Sigmoid:y:0,lstm_8/while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22!
lstm_8/while/lstm_cell_10/mul_1?
lstm_8/while/lstm_cell_10/add_1AddV2!lstm_8/while/lstm_cell_10/mul:z:0#lstm_8/while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22!
lstm_8/while/lstm_cell_10/add_1?
#lstm_8/while/lstm_cell_10/Sigmoid_2Sigmoid(lstm_8/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????22%
#lstm_8/while/lstm_cell_10/Sigmoid_2?
 lstm_8/while/lstm_cell_10/Relu_1Relu#lstm_8/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22"
 lstm_8/while/lstm_cell_10/Relu_1?
lstm_8/while/lstm_cell_10/mul_2Mul'lstm_8/while/lstm_cell_10/Sigmoid_2:y:0.lstm_8/while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22!
lstm_8/while/lstm_cell_10/mul_2?
1lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_8_while_placeholder_1lstm_8_while_placeholder#lstm_8/while/lstm_cell_10/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_8/while/TensorArrayV2Write/TensorListSetItemj
lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add/y?
lstm_8/while/addAddV2lstm_8_while_placeholderlstm_8/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/addn
lstm_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add_1/y?
lstm_8/while/add_1AddV2&lstm_8_while_lstm_8_while_loop_counterlstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/add_1s
lstm_8/while/IdentityIdentitylstm_8/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_8/while/Identity?
lstm_8/while/Identity_1Identity,lstm_8_while_lstm_8_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_8/while/Identity_1u
lstm_8/while/Identity_2Identitylstm_8/while/add:z:0*
T0*
_output_shapes
: 2
lstm_8/while/Identity_2?
lstm_8/while/Identity_3IdentityAlstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_8/while/Identity_3?
lstm_8/while/Identity_4Identity#lstm_8/while/lstm_cell_10/mul_2:z:0*
T0*'
_output_shapes
:?????????22
lstm_8/while/Identity_4?
lstm_8/while/Identity_5Identity#lstm_8/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_8/while/Identity_5"7
lstm_8_while_identitylstm_8/while/Identity:output:0";
lstm_8_while_identity_1 lstm_8/while/Identity_1:output:0";
lstm_8_while_identity_2 lstm_8/while/Identity_2:output:0";
lstm_8_while_identity_3 lstm_8/while/Identity_3:output:0";
lstm_8_while_identity_4 lstm_8/while/Identity_4:output:0";
lstm_8_while_identity_5 lstm_8/while/Identity_5:output:0"L
#lstm_8_while_lstm_8_strided_slice_1%lstm_8_while_lstm_8_strided_slice_1_0"x
9lstm_8_while_lstm_cell_10_biasadd_readvariableop_resource;lstm_8_while_lstm_cell_10_biasadd_readvariableop_resource_0"z
:lstm_8_while_lstm_cell_10_matmul_1_readvariableop_resource<lstm_8_while_lstm_cell_10_matmul_1_readvariableop_resource_0"v
8lstm_8_while_lstm_cell_10_matmul_readvariableop_resource:lstm_8_while_lstm_cell_10_matmul_readvariableop_resource_0"?
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensoralstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*Q
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
?
?
-__inference_lstm_cell_10_layer_call_fn_241142

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
GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_2364302
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
?9
?
while_body_240063
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_10_matmul_readvariableop_resource_09
5while_lstm_cell_10_matmul_1_readvariableop_resource_08
4while_lstm_cell_10_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_10_matmul_readvariableop_resource7
3while_lstm_cell_10_matmul_1_readvariableop_resource6
2while_lstm_cell_10_biasadd_readvariableop_resource??
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp?
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/MatMul?
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOp?
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/MatMul_1?
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/add?
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOp?
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/BiasAddv
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const?
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim?
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_10/split?
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid?
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid_1?
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul?
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Relu?
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul_1?
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/add_1?
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid_2?
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Relu_1?
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_2:z:0*
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
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
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
?
?
H__inference_lstm_cell_11_layer_call_and_return_conditional_losses_241192

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
?0
?
H__inference_sequential_5_layer_call_and_return_conditional_losses_238397
time_distributed_input
time_distributed_238357
time_distributed_238359
time_distributed_1_238364
time_distributed_1_238366
lstm_8_238377
lstm_8_238379
lstm_8_238381
lstm_9_238384
lstm_9_238386
lstm_9_238388
dense_9_238391
dense_9_238393
identity??dense_9/StatefulPartitionedCall?lstm_8/StatefulPartitionedCall?lstm_9/StatefulPartitionedCall?(time_distributed/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?
(time_distributed/StatefulPartitionedCallStatefulPartitionedCalltime_distributed_inputtime_distributed_238357time_distributed_238359*
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
L__inference_time_distributed_layer_call_and_return_conditional_losses_2360352*
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
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_238364time_distributed_1_238366*
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
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_2361662,
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
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_2362652$
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
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_2363542$
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
lstm_8/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0lstm_8_238377lstm_8_238379lstm_8_238381*
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
B__inference_lstm_8_layer_call_and_return_conditional_losses_2379622 
lstm_8/StatefulPartitionedCall?
lstm_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0lstm_9_238384lstm_9_238386lstm_9_238388*
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
B__inference_lstm_9_layer_call_and_return_conditional_losses_2382972 
lstm_9/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_9/StatefulPartitionedCall:output:0dense_9_238391dense_9_238393*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_2383372!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_9/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall2T
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
-__inference_lstm_cell_10_layer_call_fn_241159

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
GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_2364632
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
?
?
while_cond_240718
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_240718___redundant_placeholder04
0while_while_cond_240718___redundant_placeholder14
0while_while_cond_240718___redundant_placeholder24
0while_while_cond_240718___redundant_placeholder3
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
?
?
D__inference_conv1d_3_layer_call_and_return_conditional_losses_236069

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
?_
?
__inference__traced_save_241417
file_prefix-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop6
2savev2_time_distributed_kernel_read_readvariableop4
0savev2_time_distributed_bias_read_readvariableop8
4savev2_time_distributed_1_kernel_read_readvariableop6
2savev2_time_distributed_1_bias_read_readvariableop9
5savev2_lstm_8_lstm_cell_10_kernel_read_readvariableopC
?savev2_lstm_8_lstm_cell_10_recurrent_kernel_read_readvariableop7
3savev2_lstm_8_lstm_cell_10_bias_read_readvariableop9
5savev2_lstm_9_lstm_cell_11_kernel_read_readvariableopC
?savev2_lstm_9_lstm_cell_11_recurrent_kernel_read_readvariableop7
3savev2_lstm_9_lstm_cell_11_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop=
9savev2_adam_time_distributed_kernel_m_read_readvariableop;
7savev2_adam_time_distributed_bias_m_read_readvariableop?
;savev2_adam_time_distributed_1_kernel_m_read_readvariableop=
9savev2_adam_time_distributed_1_bias_m_read_readvariableop@
<savev2_adam_lstm_8_lstm_cell_10_kernel_m_read_readvariableopJ
Fsavev2_adam_lstm_8_lstm_cell_10_recurrent_kernel_m_read_readvariableop>
:savev2_adam_lstm_8_lstm_cell_10_bias_m_read_readvariableop@
<savev2_adam_lstm_9_lstm_cell_11_kernel_m_read_readvariableopJ
Fsavev2_adam_lstm_9_lstm_cell_11_recurrent_kernel_m_read_readvariableop>
:savev2_adam_lstm_9_lstm_cell_11_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop=
9savev2_adam_time_distributed_kernel_v_read_readvariableop;
7savev2_adam_time_distributed_bias_v_read_readvariableop?
;savev2_adam_time_distributed_1_kernel_v_read_readvariableop=
9savev2_adam_time_distributed_1_bias_v_read_readvariableop@
<savev2_adam_lstm_8_lstm_cell_10_kernel_v_read_readvariableopJ
Fsavev2_adam_lstm_8_lstm_cell_10_recurrent_kernel_v_read_readvariableop>
:savev2_adam_lstm_8_lstm_cell_10_bias_v_read_readvariableop@
<savev2_adam_lstm_9_lstm_cell_11_kernel_v_read_readvariableopJ
Fsavev2_adam_lstm_9_lstm_cell_11_recurrent_kernel_v_read_readvariableop>
:savev2_adam_lstm_9_lstm_cell_11_bias_v_read_readvariableop
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
value3B1 B+_temp_9b709e02be344c36bae1afa9e86590bf/part2	
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_time_distributed_kernel_read_readvariableop0savev2_time_distributed_bias_read_readvariableop4savev2_time_distributed_1_kernel_read_readvariableop2savev2_time_distributed_1_bias_read_readvariableop5savev2_lstm_8_lstm_cell_10_kernel_read_readvariableop?savev2_lstm_8_lstm_cell_10_recurrent_kernel_read_readvariableop3savev2_lstm_8_lstm_cell_10_bias_read_readvariableop5savev2_lstm_9_lstm_cell_11_kernel_read_readvariableop?savev2_lstm_9_lstm_cell_11_recurrent_kernel_read_readvariableop3savev2_lstm_9_lstm_cell_11_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop9savev2_adam_time_distributed_kernel_m_read_readvariableop7savev2_adam_time_distributed_bias_m_read_readvariableop;savev2_adam_time_distributed_1_kernel_m_read_readvariableop9savev2_adam_time_distributed_1_bias_m_read_readvariableop<savev2_adam_lstm_8_lstm_cell_10_kernel_m_read_readvariableopFsavev2_adam_lstm_8_lstm_cell_10_recurrent_kernel_m_read_readvariableop:savev2_adam_lstm_8_lstm_cell_10_bias_m_read_readvariableop<savev2_adam_lstm_9_lstm_cell_11_kernel_m_read_readvariableopFsavev2_adam_lstm_9_lstm_cell_11_recurrent_kernel_m_read_readvariableop:savev2_adam_lstm_9_lstm_cell_11_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop9savev2_adam_time_distributed_kernel_v_read_readvariableop7savev2_adam_time_distributed_bias_v_read_readvariableop;savev2_adam_time_distributed_1_kernel_v_read_readvariableop9savev2_adam_time_distributed_1_bias_v_read_readvariableop<savev2_adam_lstm_8_lstm_cell_10_kernel_v_read_readvariableopFsavev2_adam_lstm_8_lstm_cell_10_recurrent_kernel_v_read_readvariableop:savev2_adam_lstm_8_lstm_cell_10_bias_v_read_readvariableop<savev2_adam_lstm_9_lstm_cell_11_kernel_v_read_readvariableopFsavev2_adam_lstm_9_lstm_cell_11_recurrent_kernel_v_read_readvariableop:savev2_adam_lstm_9_lstm_cell_11_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: :(:(: : : : : :@:@:@ : :
??:	2?:?:2d:d:d: : : : :(:(:@:@:@ : :
??:	2?:?:2d:d:d:(:(:@:@:@ : :
??:	2?:?:2d:d:d: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:(: 

_output_shapes
:(:
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

:(: 

_output_shapes
:(:($
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

:(: #

_output_shapes
:(:($$
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
?
-__inference_sequential_5_layer_call_fn_239425

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
:?????????(*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_2385152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

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
while_body_240216
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_10_matmul_readvariableop_resource_09
5while_lstm_cell_10_matmul_1_readvariableop_resource_08
4while_lstm_cell_10_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_10_matmul_readvariableop_resource7
3while_lstm_cell_10_matmul_1_readvariableop_resource6
2while_lstm_cell_10_biasadd_readvariableop_resource??
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp?
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/MatMul?
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOp?
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/MatMul_1?
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/add?
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOp?
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_10/BiasAddv
while/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_10/Const?
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim?
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_10/split?
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid?
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid_1?
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul?
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Relu?
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul_1?
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/add_1?
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Sigmoid_2?
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/Relu_1?
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_10/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_2:z:0*
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
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
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
while_cond_240543
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_240543___redundant_placeholder04
0while_while_cond_240543___redundant_placeholder14
0while_while_cond_240543___redundant_placeholder24
0while_while_cond_240543___redundant_placeholder3
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
'__inference_lstm_8_layer_call_fn_240312

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
B__inference_lstm_8_layer_call_and_return_conditional_losses_2378092
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
?
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_239593

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
?
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_236354

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
E__inference_flatten_1_layer_call_and_return_conditional_losses_2362852
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
?
O
3__inference_time_distributed_3_layer_call_fn_239667

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
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_2363542
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
??
?
!__inference__wrapped_model_235911
time_distributed_inputV
Rsequential_5_time_distributed_conv1d_2_conv1d_expanddims_1_readvariableop_resourceJ
Fsequential_5_time_distributed_conv1d_2_biasadd_readvariableop_resourceX
Tsequential_5_time_distributed_1_conv1d_3_conv1d_expanddims_1_readvariableop_resourceL
Hsequential_5_time_distributed_1_conv1d_3_biasadd_readvariableop_resourceC
?sequential_5_lstm_8_lstm_cell_10_matmul_readvariableop_resourceE
Asequential_5_lstm_8_lstm_cell_10_matmul_1_readvariableop_resourceD
@sequential_5_lstm_8_lstm_cell_10_biasadd_readvariableop_resourceC
?sequential_5_lstm_9_lstm_cell_11_matmul_readvariableop_resourceE
Asequential_5_lstm_9_lstm_cell_11_matmul_1_readvariableop_resourceD
@sequential_5_lstm_9_lstm_cell_11_biasadd_readvariableop_resource7
3sequential_5_dense_9_matmul_readvariableop_resource8
4sequential_5_dense_9_biasadd_readvariableop_resource
identity??sequential_5/lstm_8/while?sequential_5/lstm_9/while?
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
sequential_5/lstm_8/ShapeShape2sequential_5/time_distributed_3/Reshape_1:output:0*
T0*
_output_shapes
:2
sequential_5/lstm_8/Shape?
'sequential_5/lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/lstm_8/strided_slice/stack?
)sequential_5/lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_5/lstm_8/strided_slice/stack_1?
)sequential_5/lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_5/lstm_8/strided_slice/stack_2?
!sequential_5/lstm_8/strided_sliceStridedSlice"sequential_5/lstm_8/Shape:output:00sequential_5/lstm_8/strided_slice/stack:output:02sequential_5/lstm_8/strided_slice/stack_1:output:02sequential_5/lstm_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_5/lstm_8/strided_slice?
sequential_5/lstm_8/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
sequential_5/lstm_8/zeros/mul/y?
sequential_5/lstm_8/zeros/mulMul*sequential_5/lstm_8/strided_slice:output:0(sequential_5/lstm_8/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_5/lstm_8/zeros/mul?
 sequential_5/lstm_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_5/lstm_8/zeros/Less/y?
sequential_5/lstm_8/zeros/LessLess!sequential_5/lstm_8/zeros/mul:z:0)sequential_5/lstm_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_5/lstm_8/zeros/Less?
"sequential_5/lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"sequential_5/lstm_8/zeros/packed/1?
 sequential_5/lstm_8/zeros/packedPack*sequential_5/lstm_8/strided_slice:output:0+sequential_5/lstm_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_5/lstm_8/zeros/packed?
sequential_5/lstm_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_5/lstm_8/zeros/Const?
sequential_5/lstm_8/zerosFill)sequential_5/lstm_8/zeros/packed:output:0(sequential_5/lstm_8/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
sequential_5/lstm_8/zeros?
!sequential_5/lstm_8/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22#
!sequential_5/lstm_8/zeros_1/mul/y?
sequential_5/lstm_8/zeros_1/mulMul*sequential_5/lstm_8/strided_slice:output:0*sequential_5/lstm_8/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_5/lstm_8/zeros_1/mul?
"sequential_5/lstm_8/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_5/lstm_8/zeros_1/Less/y?
 sequential_5/lstm_8/zeros_1/LessLess#sequential_5/lstm_8/zeros_1/mul:z:0+sequential_5/lstm_8/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_5/lstm_8/zeros_1/Less?
$sequential_5/lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22&
$sequential_5/lstm_8/zeros_1/packed/1?
"sequential_5/lstm_8/zeros_1/packedPack*sequential_5/lstm_8/strided_slice:output:0-sequential_5/lstm_8/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_5/lstm_8/zeros_1/packed?
!sequential_5/lstm_8/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_5/lstm_8/zeros_1/Const?
sequential_5/lstm_8/zeros_1Fill+sequential_5/lstm_8/zeros_1/packed:output:0*sequential_5/lstm_8/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
sequential_5/lstm_8/zeros_1?
"sequential_5/lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_5/lstm_8/transpose/perm?
sequential_5/lstm_8/transpose	Transpose2sequential_5/time_distributed_3/Reshape_1:output:0+sequential_5/lstm_8/transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
sequential_5/lstm_8/transpose?
sequential_5/lstm_8/Shape_1Shape!sequential_5/lstm_8/transpose:y:0*
T0*
_output_shapes
:2
sequential_5/lstm_8/Shape_1?
)sequential_5/lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_5/lstm_8/strided_slice_1/stack?
+sequential_5/lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_8/strided_slice_1/stack_1?
+sequential_5/lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_8/strided_slice_1/stack_2?
#sequential_5/lstm_8/strided_slice_1StridedSlice$sequential_5/lstm_8/Shape_1:output:02sequential_5/lstm_8/strided_slice_1/stack:output:04sequential_5/lstm_8/strided_slice_1/stack_1:output:04sequential_5/lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_5/lstm_8/strided_slice_1?
/sequential_5/lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/sequential_5/lstm_8/TensorArrayV2/element_shape?
!sequential_5/lstm_8/TensorArrayV2TensorListReserve8sequential_5/lstm_8/TensorArrayV2/element_shape:output:0,sequential_5/lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_5/lstm_8/TensorArrayV2?
Isequential_5/lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2K
Isequential_5/lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape?
;sequential_5/lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_5/lstm_8/transpose:y:0Rsequential_5/lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_5/lstm_8/TensorArrayUnstack/TensorListFromTensor?
)sequential_5/lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_5/lstm_8/strided_slice_2/stack?
+sequential_5/lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_8/strided_slice_2/stack_1?
+sequential_5/lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_8/strided_slice_2/stack_2?
#sequential_5/lstm_8/strided_slice_2StridedSlice!sequential_5/lstm_8/transpose:y:02sequential_5/lstm_8/strided_slice_2/stack:output:04sequential_5/lstm_8/strided_slice_2/stack_1:output:04sequential_5/lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2%
#sequential_5/lstm_8/strided_slice_2?
6sequential_5/lstm_8/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp?sequential_5_lstm_8_lstm_cell_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6sequential_5/lstm_8/lstm_cell_10/MatMul/ReadVariableOp?
'sequential_5/lstm_8/lstm_cell_10/MatMulMatMul,sequential_5/lstm_8/strided_slice_2:output:0>sequential_5/lstm_8/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_5/lstm_8/lstm_cell_10/MatMul?
8sequential_5/lstm_8/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOpAsequential_5_lstm_8_lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02:
8sequential_5/lstm_8/lstm_cell_10/MatMul_1/ReadVariableOp?
)sequential_5/lstm_8/lstm_cell_10/MatMul_1MatMul"sequential_5/lstm_8/zeros:output:0@sequential_5/lstm_8/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_5/lstm_8/lstm_cell_10/MatMul_1?
$sequential_5/lstm_8/lstm_cell_10/addAddV21sequential_5/lstm_8/lstm_cell_10/MatMul:product:03sequential_5/lstm_8/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2&
$sequential_5/lstm_8/lstm_cell_10/add?
7sequential_5/lstm_8/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp@sequential_5_lstm_8_lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7sequential_5/lstm_8/lstm_cell_10/BiasAdd/ReadVariableOp?
(sequential_5/lstm_8/lstm_cell_10/BiasAddBiasAdd(sequential_5/lstm_8/lstm_cell_10/add:z:0?sequential_5/lstm_8/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(sequential_5/lstm_8/lstm_cell_10/BiasAdd?
&sequential_5/lstm_8/lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_5/lstm_8/lstm_cell_10/Const?
0sequential_5/lstm_8/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential_5/lstm_8/lstm_cell_10/split/split_dim?
&sequential_5/lstm_8/lstm_cell_10/splitSplit9sequential_5/lstm_8/lstm_cell_10/split/split_dim:output:01sequential_5/lstm_8/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2(
&sequential_5/lstm_8/lstm_cell_10/split?
(sequential_5/lstm_8/lstm_cell_10/SigmoidSigmoid/sequential_5/lstm_8/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????22*
(sequential_5/lstm_8/lstm_cell_10/Sigmoid?
*sequential_5/lstm_8/lstm_cell_10/Sigmoid_1Sigmoid/sequential_5/lstm_8/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????22,
*sequential_5/lstm_8/lstm_cell_10/Sigmoid_1?
$sequential_5/lstm_8/lstm_cell_10/mulMul.sequential_5/lstm_8/lstm_cell_10/Sigmoid_1:y:0$sequential_5/lstm_8/zeros_1:output:0*
T0*'
_output_shapes
:?????????22&
$sequential_5/lstm_8/lstm_cell_10/mul?
%sequential_5/lstm_8/lstm_cell_10/ReluRelu/sequential_5/lstm_8/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22'
%sequential_5/lstm_8/lstm_cell_10/Relu?
&sequential_5/lstm_8/lstm_cell_10/mul_1Mul,sequential_5/lstm_8/lstm_cell_10/Sigmoid:y:03sequential_5/lstm_8/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22(
&sequential_5/lstm_8/lstm_cell_10/mul_1?
&sequential_5/lstm_8/lstm_cell_10/add_1AddV2(sequential_5/lstm_8/lstm_cell_10/mul:z:0*sequential_5/lstm_8/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22(
&sequential_5/lstm_8/lstm_cell_10/add_1?
*sequential_5/lstm_8/lstm_cell_10/Sigmoid_2Sigmoid/sequential_5/lstm_8/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????22,
*sequential_5/lstm_8/lstm_cell_10/Sigmoid_2?
'sequential_5/lstm_8/lstm_cell_10/Relu_1Relu*sequential_5/lstm_8/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22)
'sequential_5/lstm_8/lstm_cell_10/Relu_1?
&sequential_5/lstm_8/lstm_cell_10/mul_2Mul.sequential_5/lstm_8/lstm_cell_10/Sigmoid_2:y:05sequential_5/lstm_8/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22(
&sequential_5/lstm_8/lstm_cell_10/mul_2?
1sequential_5/lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   23
1sequential_5/lstm_8/TensorArrayV2_1/element_shape?
#sequential_5/lstm_8/TensorArrayV2_1TensorListReserve:sequential_5/lstm_8/TensorArrayV2_1/element_shape:output:0,sequential_5/lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_5/lstm_8/TensorArrayV2_1v
sequential_5/lstm_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_5/lstm_8/time?
,sequential_5/lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,sequential_5/lstm_8/while/maximum_iterations?
&sequential_5/lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_5/lstm_8/while/loop_counter?
sequential_5/lstm_8/whileWhile/sequential_5/lstm_8/while/loop_counter:output:05sequential_5/lstm_8/while/maximum_iterations:output:0!sequential_5/lstm_8/time:output:0,sequential_5/lstm_8/TensorArrayV2_1:handle:0"sequential_5/lstm_8/zeros:output:0$sequential_5/lstm_8/zeros_1:output:0,sequential_5/lstm_8/strided_slice_1:output:0Ksequential_5/lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_5_lstm_8_lstm_cell_10_matmul_readvariableop_resourceAsequential_5_lstm_8_lstm_cell_10_matmul_1_readvariableop_resource@sequential_5_lstm_8_lstm_cell_10_biasadd_readvariableop_resource*
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
%sequential_5_lstm_8_while_body_235671*1
cond)R'
%sequential_5_lstm_8_while_cond_235670*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
sequential_5/lstm_8/while?
Dsequential_5/lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2F
Dsequential_5/lstm_8/TensorArrayV2Stack/TensorListStack/element_shape?
6sequential_5/lstm_8/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_5/lstm_8/while:output:3Msequential_5/lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype028
6sequential_5/lstm_8/TensorArrayV2Stack/TensorListStack?
)sequential_5/lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)sequential_5/lstm_8/strided_slice_3/stack?
+sequential_5/lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_5/lstm_8/strided_slice_3/stack_1?
+sequential_5/lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_8/strided_slice_3/stack_2?
#sequential_5/lstm_8/strided_slice_3StridedSlice?sequential_5/lstm_8/TensorArrayV2Stack/TensorListStack:tensor:02sequential_5/lstm_8/strided_slice_3/stack:output:04sequential_5/lstm_8/strided_slice_3/stack_1:output:04sequential_5/lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2%
#sequential_5/lstm_8/strided_slice_3?
$sequential_5/lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_5/lstm_8/transpose_1/perm?
sequential_5/lstm_8/transpose_1	Transpose?sequential_5/lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_5/lstm_8/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22!
sequential_5/lstm_8/transpose_1?
sequential_5/lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_5/lstm_8/runtime?
sequential_5/lstm_9/ShapeShape#sequential_5/lstm_8/transpose_1:y:0*
T0*
_output_shapes
:2
sequential_5/lstm_9/Shape?
'sequential_5/lstm_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/lstm_9/strided_slice/stack?
)sequential_5/lstm_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_5/lstm_9/strided_slice/stack_1?
)sequential_5/lstm_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_5/lstm_9/strided_slice/stack_2?
!sequential_5/lstm_9/strided_sliceStridedSlice"sequential_5/lstm_9/Shape:output:00sequential_5/lstm_9/strided_slice/stack:output:02sequential_5/lstm_9/strided_slice/stack_1:output:02sequential_5/lstm_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_5/lstm_9/strided_slice?
sequential_5/lstm_9/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_5/lstm_9/zeros/mul/y?
sequential_5/lstm_9/zeros/mulMul*sequential_5/lstm_9/strided_slice:output:0(sequential_5/lstm_9/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_5/lstm_9/zeros/mul?
 sequential_5/lstm_9/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_5/lstm_9/zeros/Less/y?
sequential_5/lstm_9/zeros/LessLess!sequential_5/lstm_9/zeros/mul:z:0)sequential_5/lstm_9/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_5/lstm_9/zeros/Less?
"sequential_5/lstm_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_5/lstm_9/zeros/packed/1?
 sequential_5/lstm_9/zeros/packedPack*sequential_5/lstm_9/strided_slice:output:0+sequential_5/lstm_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_5/lstm_9/zeros/packed?
sequential_5/lstm_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_5/lstm_9/zeros/Const?
sequential_5/lstm_9/zerosFill)sequential_5/lstm_9/zeros/packed:output:0(sequential_5/lstm_9/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
sequential_5/lstm_9/zeros?
!sequential_5/lstm_9/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_5/lstm_9/zeros_1/mul/y?
sequential_5/lstm_9/zeros_1/mulMul*sequential_5/lstm_9/strided_slice:output:0*sequential_5/lstm_9/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_5/lstm_9/zeros_1/mul?
"sequential_5/lstm_9/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_5/lstm_9/zeros_1/Less/y?
 sequential_5/lstm_9/zeros_1/LessLess#sequential_5/lstm_9/zeros_1/mul:z:0+sequential_5/lstm_9/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_5/lstm_9/zeros_1/Less?
$sequential_5/lstm_9/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$sequential_5/lstm_9/zeros_1/packed/1?
"sequential_5/lstm_9/zeros_1/packedPack*sequential_5/lstm_9/strided_slice:output:0-sequential_5/lstm_9/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_5/lstm_9/zeros_1/packed?
!sequential_5/lstm_9/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_5/lstm_9/zeros_1/Const?
sequential_5/lstm_9/zeros_1Fill+sequential_5/lstm_9/zeros_1/packed:output:0*sequential_5/lstm_9/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
sequential_5/lstm_9/zeros_1?
"sequential_5/lstm_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_5/lstm_9/transpose/perm?
sequential_5/lstm_9/transpose	Transpose#sequential_5/lstm_8/transpose_1:y:0+sequential_5/lstm_9/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
sequential_5/lstm_9/transpose?
sequential_5/lstm_9/Shape_1Shape!sequential_5/lstm_9/transpose:y:0*
T0*
_output_shapes
:2
sequential_5/lstm_9/Shape_1?
)sequential_5/lstm_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_5/lstm_9/strided_slice_1/stack?
+sequential_5/lstm_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_9/strided_slice_1/stack_1?
+sequential_5/lstm_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_9/strided_slice_1/stack_2?
#sequential_5/lstm_9/strided_slice_1StridedSlice$sequential_5/lstm_9/Shape_1:output:02sequential_5/lstm_9/strided_slice_1/stack:output:04sequential_5/lstm_9/strided_slice_1/stack_1:output:04sequential_5/lstm_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_5/lstm_9/strided_slice_1?
/sequential_5/lstm_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/sequential_5/lstm_9/TensorArrayV2/element_shape?
!sequential_5/lstm_9/TensorArrayV2TensorListReserve8sequential_5/lstm_9/TensorArrayV2/element_shape:output:0,sequential_5/lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_5/lstm_9/TensorArrayV2?
Isequential_5/lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2K
Isequential_5/lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape?
;sequential_5/lstm_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_5/lstm_9/transpose:y:0Rsequential_5/lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_5/lstm_9/TensorArrayUnstack/TensorListFromTensor?
)sequential_5/lstm_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_5/lstm_9/strided_slice_2/stack?
+sequential_5/lstm_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_9/strided_slice_2/stack_1?
+sequential_5/lstm_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_9/strided_slice_2/stack_2?
#sequential_5/lstm_9/strided_slice_2StridedSlice!sequential_5/lstm_9/transpose:y:02sequential_5/lstm_9/strided_slice_2/stack:output:04sequential_5/lstm_9/strided_slice_2/stack_1:output:04sequential_5/lstm_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2%
#sequential_5/lstm_9/strided_slice_2?
6sequential_5/lstm_9/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp?sequential_5_lstm_9_lstm_cell_11_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype028
6sequential_5/lstm_9/lstm_cell_11/MatMul/ReadVariableOp?
'sequential_5/lstm_9/lstm_cell_11/MatMulMatMul,sequential_5/lstm_9/strided_slice_2:output:0>sequential_5/lstm_9/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2)
'sequential_5/lstm_9/lstm_cell_11/MatMul?
8sequential_5/lstm_9/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOpAsequential_5_lstm_9_lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype02:
8sequential_5/lstm_9/lstm_cell_11/MatMul_1/ReadVariableOp?
)sequential_5/lstm_9/lstm_cell_11/MatMul_1MatMul"sequential_5/lstm_9/zeros:output:0@sequential_5/lstm_9/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2+
)sequential_5/lstm_9/lstm_cell_11/MatMul_1?
$sequential_5/lstm_9/lstm_cell_11/addAddV21sequential_5/lstm_9/lstm_cell_11/MatMul:product:03sequential_5/lstm_9/lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2&
$sequential_5/lstm_9/lstm_cell_11/add?
7sequential_5/lstm_9/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp@sequential_5_lstm_9_lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype029
7sequential_5/lstm_9/lstm_cell_11/BiasAdd/ReadVariableOp?
(sequential_5/lstm_9/lstm_cell_11/BiasAddBiasAdd(sequential_5/lstm_9/lstm_cell_11/add:z:0?sequential_5/lstm_9/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2*
(sequential_5/lstm_9/lstm_cell_11/BiasAdd?
&sequential_5/lstm_9/lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_5/lstm_9/lstm_cell_11/Const?
0sequential_5/lstm_9/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential_5/lstm_9/lstm_cell_11/split/split_dim?
&sequential_5/lstm_9/lstm_cell_11/splitSplit9sequential_5/lstm_9/lstm_cell_11/split/split_dim:output:01sequential_5/lstm_9/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2(
&sequential_5/lstm_9/lstm_cell_11/split?
(sequential_5/lstm_9/lstm_cell_11/SigmoidSigmoid/sequential_5/lstm_9/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????2*
(sequential_5/lstm_9/lstm_cell_11/Sigmoid?
*sequential_5/lstm_9/lstm_cell_11/Sigmoid_1Sigmoid/sequential_5/lstm_9/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????2,
*sequential_5/lstm_9/lstm_cell_11/Sigmoid_1?
$sequential_5/lstm_9/lstm_cell_11/mulMul.sequential_5/lstm_9/lstm_cell_11/Sigmoid_1:y:0$sequential_5/lstm_9/zeros_1:output:0*
T0*'
_output_shapes
:?????????2&
$sequential_5/lstm_9/lstm_cell_11/mul?
%sequential_5/lstm_9/lstm_cell_11/ReluRelu/sequential_5/lstm_9/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2'
%sequential_5/lstm_9/lstm_cell_11/Relu?
&sequential_5/lstm_9/lstm_cell_11/mul_1Mul,sequential_5/lstm_9/lstm_cell_11/Sigmoid:y:03sequential_5/lstm_9/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2(
&sequential_5/lstm_9/lstm_cell_11/mul_1?
&sequential_5/lstm_9/lstm_cell_11/add_1AddV2(sequential_5/lstm_9/lstm_cell_11/mul:z:0*sequential_5/lstm_9/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2(
&sequential_5/lstm_9/lstm_cell_11/add_1?
*sequential_5/lstm_9/lstm_cell_11/Sigmoid_2Sigmoid/sequential_5/lstm_9/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????2,
*sequential_5/lstm_9/lstm_cell_11/Sigmoid_2?
'sequential_5/lstm_9/lstm_cell_11/Relu_1Relu*sequential_5/lstm_9/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2)
'sequential_5/lstm_9/lstm_cell_11/Relu_1?
&sequential_5/lstm_9/lstm_cell_11/mul_2Mul.sequential_5/lstm_9/lstm_cell_11/Sigmoid_2:y:05sequential_5/lstm_9/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2(
&sequential_5/lstm_9/lstm_cell_11/mul_2?
1sequential_5/lstm_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1sequential_5/lstm_9/TensorArrayV2_1/element_shape?
#sequential_5/lstm_9/TensorArrayV2_1TensorListReserve:sequential_5/lstm_9/TensorArrayV2_1/element_shape:output:0,sequential_5/lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_5/lstm_9/TensorArrayV2_1v
sequential_5/lstm_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_5/lstm_9/time?
,sequential_5/lstm_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,sequential_5/lstm_9/while/maximum_iterations?
&sequential_5/lstm_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_5/lstm_9/while/loop_counter?
sequential_5/lstm_9/whileWhile/sequential_5/lstm_9/while/loop_counter:output:05sequential_5/lstm_9/while/maximum_iterations:output:0!sequential_5/lstm_9/time:output:0,sequential_5/lstm_9/TensorArrayV2_1:handle:0"sequential_5/lstm_9/zeros:output:0$sequential_5/lstm_9/zeros_1:output:0,sequential_5/lstm_9/strided_slice_1:output:0Ksequential_5/lstm_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_5_lstm_9_lstm_cell_11_matmul_readvariableop_resourceAsequential_5_lstm_9_lstm_cell_11_matmul_1_readvariableop_resource@sequential_5_lstm_9_lstm_cell_11_biasadd_readvariableop_resource*
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
%sequential_5_lstm_9_while_body_235820*1
cond)R'
%sequential_5_lstm_9_while_cond_235819*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
sequential_5/lstm_9/while?
Dsequential_5/lstm_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dsequential_5/lstm_9/TensorArrayV2Stack/TensorListStack/element_shape?
6sequential_5/lstm_9/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_5/lstm_9/while:output:3Msequential_5/lstm_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype028
6sequential_5/lstm_9/TensorArrayV2Stack/TensorListStack?
)sequential_5/lstm_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)sequential_5/lstm_9/strided_slice_3/stack?
+sequential_5/lstm_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_5/lstm_9/strided_slice_3/stack_1?
+sequential_5/lstm_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_5/lstm_9/strided_slice_3/stack_2?
#sequential_5/lstm_9/strided_slice_3StridedSlice?sequential_5/lstm_9/TensorArrayV2Stack/TensorListStack:tensor:02sequential_5/lstm_9/strided_slice_3/stack:output:04sequential_5/lstm_9/strided_slice_3/stack_1:output:04sequential_5/lstm_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2%
#sequential_5/lstm_9/strided_slice_3?
$sequential_5/lstm_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_5/lstm_9/transpose_1/perm?
sequential_5/lstm_9/transpose_1	Transpose?sequential_5/lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_5/lstm_9/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2!
sequential_5/lstm_9/transpose_1?
sequential_5/lstm_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_5/lstm_9/runtime?
*sequential_5/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_5_dense_9_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02,
*sequential_5/dense_9/MatMul/ReadVariableOp?
sequential_5/dense_9/MatMulMatMul,sequential_5/lstm_9/strided_slice_3:output:02sequential_5/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
sequential_5/dense_9/MatMul?
+sequential_5/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_9_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02-
+sequential_5/dense_9/BiasAdd/ReadVariableOp?
sequential_5/dense_9/BiasAddBiasAdd%sequential_5/dense_9/MatMul:product:03sequential_5/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
sequential_5/dense_9/BiasAdd?
IdentityIdentity%sequential_5/dense_9/BiasAdd:output:0^sequential_5/lstm_8/while^sequential_5/lstm_9/while*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:"??????????????????
::::::::::::26
sequential_5/lstm_8/whilesequential_5/lstm_8/while26
sequential_5/lstm_9/whilesequential_5/lstm_9/while:p l
8
_output_shapes&
$:"??????????????????

0
_user_specified_nametime_distributed_input
?
?
while_cond_237498
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_237498___redundant_placeholder04
0while_while_cond_237498___redundant_placeholder14
0while_while_cond_237498___redundant_placeholder24
0while_while_cond_237498___redundant_placeholder3
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
B__inference_lstm_8_layer_call_and_return_conditional_losses_237962

inputs/
+lstm_cell_10_matmul_readvariableop_resource1
-lstm_cell_10_matmul_1_readvariableop_resource0
,lstm_cell_10_biasadd_readvariableop_resource
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOp?
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/MatMul?
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOp?
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/MatMul_1?
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/add?
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOp?
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/BiasAddj
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim?
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_10/split?
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid?
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid_1?
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul}
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Relu?
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul_1?
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/add_1?
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Relu_1?
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
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
while_body_237877*
condR
while_cond_237876*K
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
?
?
1__inference_time_distributed_layer_call_fn_239490

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
L__inference_time_distributed_layer_call_and_return_conditional_losses_2360052
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
?S
?
%sequential_5_lstm_9_while_body_235820D
@sequential_5_lstm_9_while_sequential_5_lstm_9_while_loop_counterJ
Fsequential_5_lstm_9_while_sequential_5_lstm_9_while_maximum_iterations)
%sequential_5_lstm_9_while_placeholder+
'sequential_5_lstm_9_while_placeholder_1+
'sequential_5_lstm_9_while_placeholder_2+
'sequential_5_lstm_9_while_placeholder_3C
?sequential_5_lstm_9_while_sequential_5_lstm_9_strided_slice_1_0
{sequential_5_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_9_tensorarrayunstack_tensorlistfromtensor_0K
Gsequential_5_lstm_9_while_lstm_cell_11_matmul_readvariableop_resource_0M
Isequential_5_lstm_9_while_lstm_cell_11_matmul_1_readvariableop_resource_0L
Hsequential_5_lstm_9_while_lstm_cell_11_biasadd_readvariableop_resource_0&
"sequential_5_lstm_9_while_identity(
$sequential_5_lstm_9_while_identity_1(
$sequential_5_lstm_9_while_identity_2(
$sequential_5_lstm_9_while_identity_3(
$sequential_5_lstm_9_while_identity_4(
$sequential_5_lstm_9_while_identity_5A
=sequential_5_lstm_9_while_sequential_5_lstm_9_strided_slice_1}
ysequential_5_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_9_tensorarrayunstack_tensorlistfromtensorI
Esequential_5_lstm_9_while_lstm_cell_11_matmul_readvariableop_resourceK
Gsequential_5_lstm_9_while_lstm_cell_11_matmul_1_readvariableop_resourceJ
Fsequential_5_lstm_9_while_lstm_cell_11_biasadd_readvariableop_resource??
Ksequential_5/lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2M
Ksequential_5/lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=sequential_5/lstm_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_5_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_9_tensorarrayunstack_tensorlistfromtensor_0%sequential_5_lstm_9_while_placeholderTsequential_5/lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02?
=sequential_5/lstm_9/while/TensorArrayV2Read/TensorListGetItem?
<sequential_5/lstm_9/while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOpGsequential_5_lstm_9_while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype02>
<sequential_5/lstm_9/while/lstm_cell_11/MatMul/ReadVariableOp?
-sequential_5/lstm_9/while/lstm_cell_11/MatMulMatMulDsequential_5/lstm_9/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential_5/lstm_9/while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2/
-sequential_5/lstm_9/while/lstm_cell_11/MatMul?
>sequential_5/lstm_9/while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOpIsequential_5_lstm_9_while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype02@
>sequential_5/lstm_9/while/lstm_cell_11/MatMul_1/ReadVariableOp?
/sequential_5/lstm_9/while/lstm_cell_11/MatMul_1MatMul'sequential_5_lstm_9_while_placeholder_2Fsequential_5/lstm_9/while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d21
/sequential_5/lstm_9/while/lstm_cell_11/MatMul_1?
*sequential_5/lstm_9/while/lstm_cell_11/addAddV27sequential_5/lstm_9/while/lstm_cell_11/MatMul:product:09sequential_5/lstm_9/while/lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2,
*sequential_5/lstm_9/while/lstm_cell_11/add?
=sequential_5/lstm_9/while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOpHsequential_5_lstm_9_while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype02?
=sequential_5/lstm_9/while/lstm_cell_11/BiasAdd/ReadVariableOp?
.sequential_5/lstm_9/while/lstm_cell_11/BiasAddBiasAdd.sequential_5/lstm_9/while/lstm_cell_11/add:z:0Esequential_5/lstm_9/while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d20
.sequential_5/lstm_9/while/lstm_cell_11/BiasAdd?
,sequential_5/lstm_9/while/lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_5/lstm_9/while/lstm_cell_11/Const?
6sequential_5/lstm_9/while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential_5/lstm_9/while/lstm_cell_11/split/split_dim?
,sequential_5/lstm_9/while/lstm_cell_11/splitSplit?sequential_5/lstm_9/while/lstm_cell_11/split/split_dim:output:07sequential_5/lstm_9/while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2.
,sequential_5/lstm_9/while/lstm_cell_11/split?
.sequential_5/lstm_9/while/lstm_cell_11/SigmoidSigmoid5sequential_5/lstm_9/while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????20
.sequential_5/lstm_9/while/lstm_cell_11/Sigmoid?
0sequential_5/lstm_9/while/lstm_cell_11/Sigmoid_1Sigmoid5sequential_5/lstm_9/while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????22
0sequential_5/lstm_9/while/lstm_cell_11/Sigmoid_1?
*sequential_5/lstm_9/while/lstm_cell_11/mulMul4sequential_5/lstm_9/while/lstm_cell_11/Sigmoid_1:y:0'sequential_5_lstm_9_while_placeholder_3*
T0*'
_output_shapes
:?????????2,
*sequential_5/lstm_9/while/lstm_cell_11/mul?
+sequential_5/lstm_9/while/lstm_cell_11/ReluRelu5sequential_5/lstm_9/while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2-
+sequential_5/lstm_9/while/lstm_cell_11/Relu?
,sequential_5/lstm_9/while/lstm_cell_11/mul_1Mul2sequential_5/lstm_9/while/lstm_cell_11/Sigmoid:y:09sequential_5/lstm_9/while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2.
,sequential_5/lstm_9/while/lstm_cell_11/mul_1?
,sequential_5/lstm_9/while/lstm_cell_11/add_1AddV2.sequential_5/lstm_9/while/lstm_cell_11/mul:z:00sequential_5/lstm_9/while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2.
,sequential_5/lstm_9/while/lstm_cell_11/add_1?
0sequential_5/lstm_9/while/lstm_cell_11/Sigmoid_2Sigmoid5sequential_5/lstm_9/while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????22
0sequential_5/lstm_9/while/lstm_cell_11/Sigmoid_2?
-sequential_5/lstm_9/while/lstm_cell_11/Relu_1Relu0sequential_5/lstm_9/while/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2/
-sequential_5/lstm_9/while/lstm_cell_11/Relu_1?
,sequential_5/lstm_9/while/lstm_cell_11/mul_2Mul4sequential_5/lstm_9/while/lstm_cell_11/Sigmoid_2:y:0;sequential_5/lstm_9/while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2.
,sequential_5/lstm_9/while/lstm_cell_11/mul_2?
>sequential_5/lstm_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_5_lstm_9_while_placeholder_1%sequential_5_lstm_9_while_placeholder0sequential_5/lstm_9/while/lstm_cell_11/mul_2:z:0*
_output_shapes
: *
element_dtype02@
>sequential_5/lstm_9/while/TensorArrayV2Write/TensorListSetItem?
sequential_5/lstm_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_5/lstm_9/while/add/y?
sequential_5/lstm_9/while/addAddV2%sequential_5_lstm_9_while_placeholder(sequential_5/lstm_9/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_5/lstm_9/while/add?
!sequential_5/lstm_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_5/lstm_9/while/add_1/y?
sequential_5/lstm_9/while/add_1AddV2@sequential_5_lstm_9_while_sequential_5_lstm_9_while_loop_counter*sequential_5/lstm_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_5/lstm_9/while/add_1?
"sequential_5/lstm_9/while/IdentityIdentity#sequential_5/lstm_9/while/add_1:z:0*
T0*
_output_shapes
: 2$
"sequential_5/lstm_9/while/Identity?
$sequential_5/lstm_9/while/Identity_1IdentityFsequential_5_lstm_9_while_sequential_5_lstm_9_while_maximum_iterations*
T0*
_output_shapes
: 2&
$sequential_5/lstm_9/while/Identity_1?
$sequential_5/lstm_9/while/Identity_2Identity!sequential_5/lstm_9/while/add:z:0*
T0*
_output_shapes
: 2&
$sequential_5/lstm_9/while/Identity_2?
$sequential_5/lstm_9/while/Identity_3IdentityNsequential_5/lstm_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2&
$sequential_5/lstm_9/while/Identity_3?
$sequential_5/lstm_9/while/Identity_4Identity0sequential_5/lstm_9/while/lstm_cell_11/mul_2:z:0*
T0*'
_output_shapes
:?????????2&
$sequential_5/lstm_9/while/Identity_4?
$sequential_5/lstm_9/while/Identity_5Identity0sequential_5/lstm_9/while/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2&
$sequential_5/lstm_9/while/Identity_5"Q
"sequential_5_lstm_9_while_identity+sequential_5/lstm_9/while/Identity:output:0"U
$sequential_5_lstm_9_while_identity_1-sequential_5/lstm_9/while/Identity_1:output:0"U
$sequential_5_lstm_9_while_identity_2-sequential_5/lstm_9/while/Identity_2:output:0"U
$sequential_5_lstm_9_while_identity_3-sequential_5/lstm_9/while/Identity_3:output:0"U
$sequential_5_lstm_9_while_identity_4-sequential_5/lstm_9/while/Identity_4:output:0"U
$sequential_5_lstm_9_while_identity_5-sequential_5/lstm_9/while/Identity_5:output:0"?
Fsequential_5_lstm_9_while_lstm_cell_11_biasadd_readvariableop_resourceHsequential_5_lstm_9_while_lstm_cell_11_biasadd_readvariableop_resource_0"?
Gsequential_5_lstm_9_while_lstm_cell_11_matmul_1_readvariableop_resourceIsequential_5_lstm_9_while_lstm_cell_11_matmul_1_readvariableop_resource_0"?
Esequential_5_lstm_9_while_lstm_cell_11_matmul_readvariableop_resourceGsequential_5_lstm_9_while_lstm_cell_11_matmul_readvariableop_resource_0"?
=sequential_5_lstm_9_while_sequential_5_lstm_9_strided_slice_1?sequential_5_lstm_9_while_sequential_5_lstm_9_strided_slice_1_0"?
ysequential_5_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_9_tensorarrayunstack_tensorlistfromtensor{sequential_5_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_9_tensorarrayunstack_tensorlistfromtensor_0*Q
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
?
?
'__inference_lstm_9_layer_call_fn_240640
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
B__inference_lstm_9_layer_call_and_return_conditional_losses_2374362
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
?D
?
B__inference_lstm_8_layer_call_and_return_conditional_losses_236958

inputs
lstm_cell_10_236876
lstm_cell_10_236878
lstm_cell_10_236880
identity??$lstm_cell_10/StatefulPartitionedCall?whileD
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
$lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10_236876lstm_cell_10_236878lstm_cell_10_236880*
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
GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_2364632&
$lstm_cell_10/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10_236876lstm_cell_10_236878lstm_cell_10_236880*
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
while_body_236889*
condR
while_cond_236888*K
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
IdentityIdentitytranspose_1:y:0%^lstm_cell_10/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2L
$lstm_cell_10/StatefulPartitionedCall$lstm_cell_10/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
C__inference_dense_9_layer_call_and_return_conditional_losses_238337

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
$__inference_signature_wrapper_238581
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
:?????????(*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_2359112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

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
3__inference_time_distributed_2_layer_call_fn_239618

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
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_2362432
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
?	
?
-__inference_sequential_5_layer_call_fn_239396

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
:?????????(*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_2384432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

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
?
?
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_241125

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
?X
?
B__inference_lstm_8_layer_call_and_return_conditional_losses_239973
inputs_0/
+lstm_cell_10_matmul_readvariableop_resource1
-lstm_cell_10_matmul_1_readvariableop_resource0
,lstm_cell_10_biasadd_readvariableop_resource
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOp?
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/MatMul?
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOp?
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/MatMul_1?
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/add?
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOp?
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_10/BiasAddj
lstm_cell_10/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/Const~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim?
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_10/split?
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid?
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid_1?
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul}
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Relu?
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul_1?
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/add_1?
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/Relu_1?
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_10/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
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
while_body_239888*
condR
while_cond_239887*K
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
while_body_240544
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_11_matmul_readvariableop_resource_09
5while_lstm_cell_11_matmul_1_readvariableop_resource_08
4while_lstm_cell_11_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_11_matmul_readvariableop_resource7
3while_lstm_cell_11_matmul_1_readvariableop_resource6
2while_lstm_cell_11_biasadd_readvariableop_resource??
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
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes

:2d*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp?
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/MatMul?
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOp?
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/MatMul_1?
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/add?
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOp?
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
while/lstm_cell_11/BiasAddv
while/lstm_cell_11/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_11/Const?
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dim?
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_11/split?
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid?
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid_1?
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul?
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Relu?
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul_1?
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/add_1?
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Sigmoid_2?
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/Relu_1?
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_11/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_11/mul_2:z:0*
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
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_11_biasadd_readvariableop_resource4while_lstm_cell_11_biasadd_readvariableop_resource_0"l
3while_lstm_cell_11_matmul_1_readvariableop_resource5while_lstm_cell_11_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_11_matmul_readvariableop_resource3while_lstm_cell_11_matmul_readvariableop_resource_0"0
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
?
?
H__inference_lstm_cell_11_layer_call_and_return_conditional_losses_241225

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
?
?
'__inference_lstm_9_layer_call_fn_240651
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
B__inference_lstm_9_layer_call_and_return_conditional_losses_2375682
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
?
?
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_236463

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
?D
?
B__inference_lstm_9_layer_call_and_return_conditional_losses_237568

inputs
lstm_cell_11_237486
lstm_cell_11_237488
lstm_cell_11_237490
identity??$lstm_cell_11/StatefulPartitionedCall?whileD
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
$lstm_cell_11/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_11_237486lstm_cell_11_237488lstm_cell_11_237490*
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
GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_11_layer_call_and_return_conditional_losses_2370732&
$lstm_cell_11/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_11_237486lstm_cell_11_237488lstm_cell_11_237490*
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
while_body_237499*
condR
while_cond_237498*K
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
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_11/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::2L
$lstm_cell_11/StatefulPartitionedCall$lstm_cell_11/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????2
 
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
;
dense_90
StatefulPartitionedCall:0?????????(tensorflow/serving/predict:??
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
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?P
_tf_keras_sequential?O{"class_name": "Sequential", "name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 10, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "time_distributed_input"}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 10, 1]}, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_3", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}}}, {"class_name": "LSTM", "config": {"name": "lstm_8", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTM", "config": {"name": "lstm_9", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 25, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 10, 1], "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 10, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 10, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "time_distributed_input"}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 10, 1]}, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_3", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}}}, {"class_name": "LSTM", "config": {"name": "lstm_8", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTM", "config": {"name": "lstm_9", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 25, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": ["mean_squared_error"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	layer
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_layer?
{"class_name": "TimeDistributed", "name": "time_distributed", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 10, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "time_distributed", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 10, 1]}, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 10, 1], "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 10, 1]}}
?

	layer
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "TimeDistributed", "name": "time_distributed_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "time_distributed_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 9, 64], "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 9, 64]}}
?
	layer
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TimeDistributed", "name": "time_distributed_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "time_distributed_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 8, 32], "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 8, 32]}}
?
	layer
regularization_losses
	variables
 trainable_variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TimeDistributed", "name": "time_distributed_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "time_distributed_3", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 4, 32], "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 4, 32]}}
?
"cell
#
state_spec
$regularization_losses
%	variables
&trainable_variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_8", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
?
(cell
)
state_spec
*regularization_losses
+	variables
,trainable_variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_9", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 25, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 50]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 50]}}
?

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 25}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25]}}
?
4iter

5beta_1

6beta_2
	7decay
8learning_rate.m?/m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?.v?/v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?"
	optimizer
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
	regularization_losses

	variables

Clayers
Dmetrics
Elayer_regularization_losses
trainable_variables
Flayer_metrics
Gnon_trainable_variables
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
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}}
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?
regularization_losses
	variables

Llayers
Mlayer_regularization_losses
Nlayer_metrics
trainable_variables
Ometrics
Pnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?	

;kernel
<bias
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}}
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
?
regularization_losses
	variables

Ulayers
Vlayer_regularization_losses
Wlayer_metrics
trainable_variables
Xmetrics
Ynon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
Zregularization_losses
[	variables
\trainable_variables
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
regularization_losses
	variables

^layers
_layer_regularization_losses
`layer_metrics
trainable_variables
ametrics
bnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
cregularization_losses
d	variables
etrainable_variables
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
regularization_losses
	variables

glayers
hlayer_regularization_losses
ilayer_metrics
 trainable_variables
jmetrics
knon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

=kernel
>recurrent_kernel
?bias
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_10", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
?
$regularization_losses
%	variables

players
qmetrics
rlayer_regularization_losses
&trainable_variables
slayer_metrics

tstates
unon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

@kernel
Arecurrent_kernel
Bbias
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_11", "trainable": true, "dtype": "float32", "units": 25, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
?
*regularization_losses
+	variables

zlayers
{metrics
|layer_regularization_losses
,trainable_variables
}layer_metrics

~states
non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :(2dense_9/kernel
:(2dense_9/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
0regularization_losses
1	variables
?layers
 ?layer_regularization_losses
?layer_metrics
2trainable_variables
?metrics
?non_trainable_variables
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
.:,
??2lstm_8/lstm_cell_10/kernel
7:5	2?2$lstm_8/lstm_cell_10/recurrent_kernel
':%?2lstm_8/lstm_cell_10/bias
,:*2d2lstm_9/lstm_cell_11/kernel
6:4d2$lstm_9/lstm_cell_11/recurrent_kernel
&:$d2lstm_9/lstm_cell_11/bias
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?
Hregularization_losses
I	variables
?layers
 ?layer_regularization_losses
?layer_metrics
Jtrainable_variables
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
0"
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
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
?
Qregularization_losses
R	variables
?layers
 ?layer_regularization_losses
?layer_metrics
Strainable_variables
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
0"
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
trackable_list_wrapper
?
Zregularization_losses
[	variables
?layers
 ?layer_regularization_losses
?layer_metrics
\trainable_variables
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
0"
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
trackable_list_wrapper
?
cregularization_losses
d	variables
?layers
 ?layer_regularization_losses
?layer_metrics
etrainable_variables
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
0"
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
5
=0
>1
?2"
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
?
lregularization_losses
m	variables
?layers
 ?layer_regularization_losses
?layer_metrics
ntrainable_variables
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
"0"
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
5
@0
A1
B2"
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
?
vregularization_losses
w	variables
?layers
 ?layer_regularization_losses
?layer_metrics
xtrainable_variables
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
%:#(2Adam/dense_9/kernel/m
:(2Adam/dense_9/bias/m
2:0@2Adam/time_distributed/kernel/m
(:&@2Adam/time_distributed/bias/m
4:2@ 2 Adam/time_distributed_1/kernel/m
*:( 2Adam/time_distributed_1/bias/m
3:1
??2!Adam/lstm_8/lstm_cell_10/kernel/m
<::	2?2+Adam/lstm_8/lstm_cell_10/recurrent_kernel/m
,:*?2Adam/lstm_8/lstm_cell_10/bias/m
1:/2d2!Adam/lstm_9/lstm_cell_11/kernel/m
;:9d2+Adam/lstm_9/lstm_cell_11/recurrent_kernel/m
+:)d2Adam/lstm_9/lstm_cell_11/bias/m
%:#(2Adam/dense_9/kernel/v
:(2Adam/dense_9/bias/v
2:0@2Adam/time_distributed/kernel/v
(:&@2Adam/time_distributed/bias/v
4:2@ 2 Adam/time_distributed_1/kernel/v
*:( 2Adam/time_distributed_1/bias/v
3:1
??2!Adam/lstm_8/lstm_cell_10/kernel/v
<::	2?2+Adam/lstm_8/lstm_cell_10/recurrent_kernel/v
,:*?2Adam/lstm_8/lstm_cell_10/bias/v
1:/2d2!Adam/lstm_9/lstm_cell_11/kernel/v
;:9d2+Adam/lstm_9/lstm_cell_11/recurrent_kernel/v
+:)d2Adam/lstm_9/lstm_cell_11/bias/v
?2?
H__inference_sequential_5_layer_call_and_return_conditional_losses_238974
H__inference_sequential_5_layer_call_and_return_conditional_losses_239367
H__inference_sequential_5_layer_call_and_return_conditional_losses_238354
H__inference_sequential_5_layer_call_and_return_conditional_losses_238397?
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
-__inference_sequential_5_layer_call_fn_238542
-__inference_sequential_5_layer_call_fn_239396
-__inference_sequential_5_layer_call_fn_239425
-__inference_sequential_5_layer_call_fn_238470?
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
?2?
!__inference__wrapped_model_235911?
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
?2?
L__inference_time_distributed_layer_call_and_return_conditional_losses_239481
L__inference_time_distributed_layer_call_and_return_conditional_losses_239453?
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
1__inference_time_distributed_layer_call_fn_239499
1__inference_time_distributed_layer_call_fn_239490?
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
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_239555
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_239527?
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
3__inference_time_distributed_1_layer_call_fn_239564
3__inference_time_distributed_1_layer_call_fn_239573?
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
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_239613
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_239593?
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
3__inference_time_distributed_2_layer_call_fn_239623
3__inference_time_distributed_2_layer_call_fn_239618?
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
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_239640
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_239657?
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
3__inference_time_distributed_3_layer_call_fn_239667
3__inference_time_distributed_3_layer_call_fn_239662?
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
B__inference_lstm_8_layer_call_and_return_conditional_losses_239820
B__inference_lstm_8_layer_call_and_return_conditional_losses_240148
B__inference_lstm_8_layer_call_and_return_conditional_losses_240301
B__inference_lstm_8_layer_call_and_return_conditional_losses_239973?
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
'__inference_lstm_8_layer_call_fn_240323
'__inference_lstm_8_layer_call_fn_239995
'__inference_lstm_8_layer_call_fn_240312
'__inference_lstm_8_layer_call_fn_239984?
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
B__inference_lstm_9_layer_call_and_return_conditional_losses_240629
B__inference_lstm_9_layer_call_and_return_conditional_losses_240804
B__inference_lstm_9_layer_call_and_return_conditional_losses_240957
B__inference_lstm_9_layer_call_and_return_conditional_losses_240476?
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
'__inference_lstm_9_layer_call_fn_240651
'__inference_lstm_9_layer_call_fn_240968
'__inference_lstm_9_layer_call_fn_240640
'__inference_lstm_9_layer_call_fn_240979?
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
C__inference_dense_9_layer_call_and_return_conditional_losses_240989?
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
(__inference_dense_9_layer_call_fn_240998?
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
$__inference_signature_wrapper_238581time_distributed_input
?2?
D__inference_conv1d_2_layer_call_and_return_conditional_losses_241014?
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
)__inference_conv1d_2_layer_call_fn_241023?
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
D__inference_conv1d_3_layer_call_and_return_conditional_losses_241039?
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
)__inference_conv1d_3_layer_call_fn_241048?
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
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_236182?
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
0__inference_max_pooling1d_1_layer_call_fn_236188?
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
E__inference_flatten_1_layer_call_and_return_conditional_losses_241054?
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
*__inference_flatten_1_layer_call_fn_241059?
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
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_241125
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_241092?
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
-__inference_lstm_cell_10_layer_call_fn_241159
-__inference_lstm_cell_10_layer_call_fn_241142?
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
H__inference_lstm_cell_11_layer_call_and_return_conditional_losses_241225
H__inference_lstm_cell_11_layer_call_and_return_conditional_losses_241192?
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
-__inference_lstm_cell_11_layer_call_fn_241242
-__inference_lstm_cell_11_layer_call_fn_241259?
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
!__inference__wrapped_model_235911?9:;<=>?@AB./P?M
F?C
A?>
time_distributed_input"??????????????????

? "1?.
,
dense_9!?
dense_9?????????(?
D__inference_conv1d_2_layer_call_and_return_conditional_losses_241014d9:3?0
)?&
$?!
inputs?????????

? ")?&
?
0?????????	@
? ?
)__inference_conv1d_2_layer_call_fn_241023W9:3?0
)?&
$?!
inputs?????????

? "??????????	@?
D__inference_conv1d_3_layer_call_and_return_conditional_losses_241039d;<3?0
)?&
$?!
inputs?????????	@
? ")?&
?
0????????? 
? ?
)__inference_conv1d_3_layer_call_fn_241048W;<3?0
)?&
$?!
inputs?????????	@
? "?????????? ?
C__inference_dense_9_layer_call_and_return_conditional_losses_240989\.//?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????(
? {
(__inference_dense_9_layer_call_fn_240998O.//?,
%?"
 ?
inputs?????????
? "??????????(?
E__inference_flatten_1_layer_call_and_return_conditional_losses_241054]3?0
)?&
$?!
inputs????????? 
? "&?#
?
0??????????
? ~
*__inference_flatten_1_layer_call_fn_241059P3?0
)?&
$?!
inputs????????? 
? "????????????
B__inference_lstm_8_layer_call_and_return_conditional_losses_239820?=>?P?M
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
B__inference_lstm_8_layer_call_and_return_conditional_losses_239973?=>?P?M
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
B__inference_lstm_8_layer_call_and_return_conditional_losses_240148?=>?I?F
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
B__inference_lstm_8_layer_call_and_return_conditional_losses_240301?=>?I?F
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
'__inference_lstm_8_layer_call_fn_239984~=>?P?M
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
'__inference_lstm_8_layer_call_fn_239995~=>?P?M
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
'__inference_lstm_8_layer_call_fn_240312w=>?I?F
??<
.?+
inputs???????????????????

 
p

 
? "%?"??????????????????2?
'__inference_lstm_8_layer_call_fn_240323w=>?I?F
??<
.?+
inputs???????????????????

 
p 

 
? "%?"??????????????????2?
B__inference_lstm_9_layer_call_and_return_conditional_losses_240476}@ABO?L
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
B__inference_lstm_9_layer_call_and_return_conditional_losses_240629}@ABO?L
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
B__inference_lstm_9_layer_call_and_return_conditional_losses_240804v@ABH?E
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
B__inference_lstm_9_layer_call_and_return_conditional_losses_240957v@ABH?E
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
'__inference_lstm_9_layer_call_fn_240640p@ABO?L
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
'__inference_lstm_9_layer_call_fn_240651p@ABO?L
E?B
4?1
/?,
inputs/0??????????????????2

 
p 

 
? "???????????
'__inference_lstm_9_layer_call_fn_240968i@ABH?E
>?;
-?*
inputs??????????????????2

 
p

 
? "???????????
'__inference_lstm_9_layer_call_fn_240979i@ABH?E
>?;
-?*
inputs??????????????????2

 
p 

 
? "???????????
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_241092?=>???~
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
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_241125?=>???~
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
-__inference_lstm_cell_10_layer_call_fn_241142?=>???~
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
-__inference_lstm_cell_10_layer_call_fn_241159?=>???~
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
H__inference_lstm_cell_11_layer_call_and_return_conditional_losses_241192?@AB??}
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
H__inference_lstm_cell_11_layer_call_and_return_conditional_losses_241225?@AB??}
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
-__inference_lstm_cell_11_layer_call_fn_241242?@AB??}
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
-__inference_lstm_cell_11_layer_call_fn_241259?@AB??}
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
K__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_236182?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
0__inference_max_pooling1d_1_layer_call_fn_236188wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
H__inference_sequential_5_layer_call_and_return_conditional_losses_238354?9:;<=>?@AB./X?U
N?K
A?>
time_distributed_input"??????????????????

p

 
? "%?"
?
0?????????(
? ?
H__inference_sequential_5_layer_call_and_return_conditional_losses_238397?9:;<=>?@AB./X?U
N?K
A?>
time_distributed_input"??????????????????

p 

 
? "%?"
?
0?????????(
? ?
H__inference_sequential_5_layer_call_and_return_conditional_losses_2389749:;<=>?@AB./H?E
>?;
1?.
inputs"??????????????????

p

 
? "%?"
?
0?????????(
? ?
H__inference_sequential_5_layer_call_and_return_conditional_losses_2393679:;<=>?@AB./H?E
>?;
1?.
inputs"??????????????????

p 

 
? "%?"
?
0?????????(
? ?
-__inference_sequential_5_layer_call_fn_238470?9:;<=>?@AB./X?U
N?K
A?>
time_distributed_input"??????????????????

p

 
? "??????????(?
-__inference_sequential_5_layer_call_fn_238542?9:;<=>?@AB./X?U
N?K
A?>
time_distributed_input"??????????????????

p 

 
? "??????????(?
-__inference_sequential_5_layer_call_fn_239396r9:;<=>?@AB./H?E
>?;
1?.
inputs"??????????????????

p

 
? "??????????(?
-__inference_sequential_5_layer_call_fn_239425r9:;<=>?@AB./H?E
>?;
1?.
inputs"??????????????????

p 

 
? "??????????(?
$__inference_signature_wrapper_238581?9:;<=>?@AB./j?g
? 
`?]
[
time_distributed_inputA?>
time_distributed_input"??????????????????
"1?.
,
dense_9!?
dense_9?????????(?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_239527?;<H?E
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
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_239555?;<H?E
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
3__inference_time_distributed_1_layer_call_fn_239564y;<H?E
>?;
1?.
inputs"??????????????????	@
p

 
? ")?&"?????????????????? ?
3__inference_time_distributed_1_layer_call_fn_239573y;<H?E
>?;
1?.
inputs"??????????????????	@
p 

 
? ")?&"?????????????????? ?
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_239593?H?E
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
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_239613?H?E
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
3__inference_time_distributed_2_layer_call_fn_239618uH?E
>?;
1?.
inputs"?????????????????? 
p

 
? ")?&"?????????????????? ?
3__inference_time_distributed_2_layer_call_fn_239623uH?E
>?;
1?.
inputs"?????????????????? 
p 

 
? ")?&"?????????????????? ?
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_239640H?E
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
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_239657H?E
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
3__inference_time_distributed_3_layer_call_fn_239662rH?E
>?;
1?.
inputs"?????????????????? 
p

 
? "&?#????????????????????
3__inference_time_distributed_3_layer_call_fn_239667rH?E
>?;
1?.
inputs"?????????????????? 
p 

 
? "&?#????????????????????
L__inference_time_distributed_layer_call_and_return_conditional_losses_239453?9:H?E
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
L__inference_time_distributed_layer_call_and_return_conditional_losses_239481?9:H?E
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
1__inference_time_distributed_layer_call_fn_239490y9:H?E
>?;
1?.
inputs"??????????????????

p

 
? ")?&"??????????????????	@?
1__inference_time_distributed_layer_call_fn_239499y9:H?E
>?;
1?.
inputs"??????????????????

p 

 
? ")?&"??????????????????	@