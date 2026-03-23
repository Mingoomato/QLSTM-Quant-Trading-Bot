#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_auto_comment_en.py — Add English comments to every code line
Usage: python scripts/_auto_comment_en.py <file1> <file2> ...
Strategy:
  - Lines with existing Korean comment (# 한글): append " / English" after
  - Lines with existing English-only comment: leave as-is
  - Lines with no comment: add English inline comment
"""
import re
import sys

IMPORT_MAP_EN = {
    'torch': 'PyTorch — core deep learning library',
    'nn': 'PyTorch neural network building blocks',
    'functional': 'PyTorch functional API (activations, losses)',
    'optim': 'PyTorch optimizers (weight update algorithms)',
    'numpy': 'NumPy — fast numerical array computation',
    'np': 'NumPy (numerical computation library)',
    'pandas': 'Pandas — tabular data manipulation library',
    'pd': 'Pandas (DataFrame library)',
    'math': 'Python standard math library (log, exp, trig)',
    'os': 'OS interface — file and directory operations',
    'sys': 'Python system and interpreter utilities',
    'time': 'Time measurement and sleep utilities',
    'logging': 'Python standard logging facility',
    'pathlib': 'Object-oriented filesystem path handling',
    'collections': 'Specialized container datatypes (deque, Counter)',
    'dataclasses': 'Auto-generate boilerplate for data-holding classes',
    'typing': 'Type hint annotations',
    'enum': 'Enumeration constants definition',
    'json': 'JSON serialization / deserialization',
    'copy': 'Shallow and deep copy utilities',
    'warnings': 'Warning message control',
    'abc': 'Abstract base class infrastructure',
    'threading': 'Thread-based parallelism',
    'argparse': 'Command-line argument parsing',
    'datetime': 'Date and time handling',
    'random': 'Random number generation',
    'csv': 'CSV file reading and writing',
    'tqdm': 'Progress bar for loops',
    'sklearn': 'Scikit-learn — machine learning algorithms',
    'scipy': 'SciPy — scientific computing (optimization, stats)',
    'pennylane': 'PennyLane — quantum computing / QML framework',
    'qml': 'PennyLane quantum ML module',
    'pybit': 'Bybit exchange REST/WebSocket API client',
    'textual': 'Textual — terminal UI framework',
    'rich': 'Rich — styled terminal output',
    'optuna': 'Optuna — automated hyperparameter optimization',
    'functools': 'Higher-order function utilities',
    'itertools': 'Iterator building blocks',
    'traceback': 'Exception traceback utilities',
    'deque': 'Double-ended queue data structure',
    'dataclass': 'Decorator to auto-generate class boilerplate',
    'field': 'Field descriptor for dataclass attributes',
    'asdict': 'Convert dataclass instance to dict',
}

PATTERN_COMMENTS_EN = [
    # PyTorch tensor ops
    (r'torch\.zeros',          '# Creates a zero-filled tensor'),
    (r'torch\.ones',           '# Creates a ones-filled tensor'),
    (r'torch\.tensor\(',       '# Converts Python data to a PyTorch tensor'),
    (r'torch\.cat\(',          '# Concatenates tensors along a given dimension'),
    (r'torch\.stack\(',        '# Stacks tensors along a new dimension'),
    (r'torch\.mean\(',         '# Computes the mean of tensor elements'),
    (r'torch\.sum\(',          '# Sums all tensor elements'),
    (r'torch\.max\(',          '# Finds the maximum value in a tensor'),
    (r'torch\.min\(',          '# Finds the minimum value in a tensor'),
    (r'torch\.log\(',          '# Computes natural logarithm element-wise'),
    (r'torch\.exp\(',          '# Computes exponential e^x element-wise'),
    (r'torch\.clamp\(',        '# Clamps tensor values to [min, max] range'),
    (r'torch\.argmax\(',       '# Returns index of the maximum value'),
    (r'torch\.arange\(',       '# Creates a 1-D tensor of evenly spaced integers'),
    (r'torch\.linspace\(',     '# Creates evenly spaced values between start and end'),
    (r'torch\.rand\(',         '# Creates a tensor with uniform random values in [0, 1)'),
    (r'torch\.randn\(',        '# Creates a tensor with standard normal random values'),
    (r'torch\.no_grad\(\)',    '# Disables gradient computation (inference mode)'),
    (r'torch\.save\(',         '# Saves a tensor/model to disk'),
    (r'torch\.load\(',         '# Loads a tensor/model from disk'),
    (r'torch\.einsum\(',       '# Performs Einstein summation (generalized matrix ops)'),
    (r'torch\.trace\(',        '# Computes the trace (sum of diagonal) of a matrix'),
    # Neural network layers
    (r'nn\.Linear\(',          '# Fully-connected (affine) layer: y = xW^T + b'),
    (r'nn\.LSTM\(',            '# Long Short-Term Memory recurrent layer'),
    (r'nn\.GRU\(',             '# Gated Recurrent Unit layer'),
    (r'nn\.Transformer',       '# Transformer (multi-head attention) layer'),
    (r'nn\.BatchNorm',         '# Batch Normalization layer (stabilizes training)'),
    (r'nn\.LayerNorm\(',       '# Layer Normalization (per-sample normalization)'),
    (r'nn\.Dropout\(',         '# Dropout layer — randomly zeros activations to prevent overfitting'),
    (r'nn\.Embedding\(',       '# Embedding layer — maps integer indices to dense vectors'),
    (r'nn\.Conv',              '# Convolutional layer for feature extraction'),
    (r'nn\.Parameter\(',       '# Registers a tensor as a learnable parameter'),
    (r'nn\.ModuleList\(',      '# Holds submodules in a list (tracked by PyTorch)'),
    (r'nn\.Sequential\(',      '# Chains layers sequentially'),
    # Activation functions
    (r'F\.relu\(',             '# ReLU activation: max(0, x)'),
    (r'F\.softmax\(',          '# Softmax: converts logits to probabilities summing to 1'),
    (r'F\.sigmoid\(',          '# Sigmoid: maps values to (0, 1)'),
    (r'F\.tanh\(',             '# Tanh activation: maps values to (-1, 1)'),
    (r'F\.log_softmax\(',      '# Log-softmax: numerically stable log of softmax'),
    (r'F\.gelu\(',             '# GELU activation (used in Transformers)'),
    # Loss functions
    (r'F\.mse_loss\(',         '# Mean Squared Error loss: mean((pred - target)^2)'),
    (r'F\.cross_entropy\(',    '# Cross-entropy loss for multi-class classification'),
    (r'F\.binary_cross_entropy','# Binary cross-entropy loss for 2-class problems'),
    # Tensor reshaping
    (r'\.reshape\(',           '# Reshapes tensor to a new shape'),
    (r'\.view\(',              '# Reshapes tensor (shares memory if possible)'),
    (r'\.squeeze\(',           '# Removes dimensions of size 1'),
    (r'\.unsqueeze\(',         '# Inserts a new dimension of size 1'),
    (r'\.permute\(',           '# Permutes tensor dimensions'),
    (r'\.transpose\(',         '# Swaps two tensor dimensions'),
    (r'\.contiguous\(\)',      '# Returns contiguous tensor in memory'),
    (r'\.float\(\)',           '# Casts tensor to float32'),
    (r'\.long\(\)',            '# Casts tensor to int64'),
    (r'\.bool\(\)',            '# Casts tensor to boolean'),
    (r'\.item\(\)',            '# Extracts a Python scalar from a 1-element tensor'),
    (r'\.cpu\(\)',             '# Moves tensor to CPU memory'),
    (r'\.cuda\(\)',            '# Moves tensor to GPU memory'),
    (r'\.to\(',                '# Moves tensor to specified device or dtype'),
    (r'\.detach\(\)',          '# Detaches tensor from the computation graph'),
    (r'\.numpy\(\)',           '# Converts tensor to NumPy array'),
    (r'\.clone\(\)',           '# Returns a deep copy of the tensor'),
    (r'\.shape',               '# Shape (dimensions) of the tensor/array'),
    (r'\.size\(',              '# Size of a specific tensor dimension'),
    # Backprop / optimization
    (r'\.backward\(\)',        '# Computes gradients via backpropagation'),
    (r'optimizer\.step\(\)',   '# Updates model parameters using computed gradients'),
    (r'optimizer\.zero_grad\(\)','# Resets gradients to zero before the next backward pass'),
    (r'torch\.nn\.utils\.clip_grad','# Clips gradient norm to prevent exploding gradients'),
    # NumPy
    (r'np\.zeros\(',           '# Creates a zero-filled array'),
    (r'np\.ones\(',            '# Creates an array filled with ones'),
    (r'np\.array\(',           '# Converts Python sequence to NumPy array'),
    (r'np\.log\(',             '# Computes natural logarithm element-wise'),
    (r'np\.exp\(',             '# Computes exponential element-wise'),
    (r'np\.mean\(',            '# Computes the mean value'),
    (r'np\.std\(',             '# Computes the standard deviation'),
    (r'np\.var\(',             '# Computes the variance'),
    (r'np\.max\(|np\.maximum\(','# Finds the maximum value'),
    (r'np\.min\(|np\.minimum\(','# Finds the minimum value'),
    (r'np\.sum\(',             '# Sums array elements'),
    (r'np\.abs\(',             '# Computes absolute values element-wise'),
    (r'np\.sqrt\(',            '# Computes square root element-wise'),
    (r'np\.clip\(',            '# Clips values to [min, max] range'),
    (r'np\.concatenate\(',     '# Concatenates arrays along an axis'),
    (r'np\.linalg\.eigh\(',    '# Eigendecomposition of a symmetric matrix'),
    (r'np\.linalg\.eig\(',     '# Eigendecomposition of a general matrix'),
    (r'np\.linalg\.norm\(',    '# Computes vector or matrix norm'),
    (r'np\.linalg\.lstsq\(',   '# Solves a least-squares linear equation'),
    (r'np\.diff\(',            '# Computes differences between consecutive elements'),
    (r'np\.where\(',           '# Returns elements chosen from two arrays by condition'),
    (r'np\.isnan\(',           '# Tests element-wise for NaN'),
    (r'np\.isinf\(',           '# Tests element-wise for infinity'),
    (r'np\.linspace\(',        '# Creates evenly spaced values over a specified interval'),
    (r'np\.arange\(',          '# Creates an array of evenly spaced integers'),
    (r'np\.argsort\(',         '# Returns indices that would sort the array'),
    (r'np\.sort\(',            '# Returns a sorted copy of an array'),
    (r'np\.cumsum\(',          '# Computes cumulative sum along an axis'),
    (r'np\.cumprod\(',         '# Computes cumulative product along an axis'),
    (r'np\.roll\(',            '# Rolls array elements along an axis'),
    (r'np\.cross\(',           '# Computes the cross product of two vectors'),
    (r'np\.dot\(',             '# Computes the dot product of two arrays'),
    (r'np\.outer\(',           '# Computes the outer product of two vectors'),
    (r'np\.cov\(',             '# Computes the covariance matrix'),
    (r'np\.corrcoef\(',        '# Computes the Pearson correlation coefficient matrix'),
    (r'np\.polyfit\(',         '# Fits a polynomial to data via least squares'),
    (r'np\.log10\(',           '# Computes base-10 logarithm'),
    (r'np\.power\(',           '# Raises elements to a given power'),
    (r'np\.median\(',          '# Computes the median value'),
    (r'np\.percentile\(',      '# Computes the q-th percentile of data'),
    (r'np\.random\.(normal|randn)','# Draws samples from a normal distribution'),
    (r'np\.random\.(uniform|rand)\b','# Draws samples from a uniform distribution'),
    (r'np\.nan_to_num\(',      '# Replaces NaN and Inf with finite numbers'),
    (r'np\.full\(',            '# Creates an array filled with a given value'),
    (r'np\.eye\(',             '# Creates an identity matrix'),
    # Quantum computing
    (r'qml\.RY\(',             '# Rotates qubit around Y-axis by angle theta'),
    (r'qml\.RX\(',             '# Rotates qubit around X-axis by angle theta'),
    (r'qml\.RZ\(',             '# Rotates qubit around Z-axis by angle theta'),
    (r'qml\.CNOT\(',           '# CNOT gate: flips target qubit if control is |1⟩'),
    (r'qml\.IsingZZ\(',        '# IsingZZ entangling gate between two qubits'),
    (r'qml\.Hadamard\(',       '# Hadamard gate: puts qubit in equal superposition'),
    (r'qml\.expval\(',         '# Measures the expectation value of an observable'),
    (r'qml\.device\(',         '# Initializes a quantum device (simulator or hardware)'),
    (r'qml\.qnode\(',          '# Wraps a Python function as a quantum circuit (QNode)'),
    (r'qml\.PauliZ\(',         '# Pauli-Z observable for qubit measurement'),
    (r'qml\.AngleEmbedding\(', '# Encodes classical data as rotation angles in qubits'),
    (r'qml\.BasicEntanglerLayers\(','# Adds a layer of entangling CNOT gates'),
    # Finance
    (r'log_return|log_ret\b',  '# Log-return: ln(P_t / P_{t-1}) — stationary price change'),
    (r'\batr\b',               '# ATR: Average True Range — average price volatility'),
    (r'sharpe',                '# Sharpe ratio: risk-adjusted return (mean / std)'),
    (r'drawdown|max_dd|mdd\b', '# Maximum drawdown: largest peak-to-trough decline'),
    (r'\bpnl\b|profit_loss',   '# PnL: realized profit and loss for this trade'),
    (r'tp_mult|take_profit',   '# Take-profit multiplier: exit at TP × ATR'),
    (r'sl_mult|stop_loss',     '# Stop-loss multiplier: forced exit at SL × ATR'),
    (r'\bequity\b',            '# Account equity: cash + unrealized PnL'),
    (r'funding_rate\b',        '# Funding rate: periodic payment between longs and shorts'),
    (r'open_interest\b',       '# Open interest: total number of outstanding contracts'),
    (r'hurst\b',               '# Hurst exponent: H>0.5 trending, H<0.5 mean-reverting'),
    (r'lindblad|decoherence',  '# Lindblad master equation: quantum decoherence model'),
    (r'platt|calibrat',        '# Platt scaling: converts raw logits to calibrated probabilities'),
    (r'fokker.planck|langevin','# Fokker-Planck / Langevin SDE regularizer'),
    (r'marchenko|rmt\b',       '# Marchenko-Pastur RMT denoising threshold'),
    (r'wasserstein',           '# Wasserstein distance: optimal transport between distributions'),
    (r'advantage|gae\b',       '# Generalized Advantage Estimation (GAE)'),
    (r'critic|value.fn',       '# Critic: estimates state-value function V(s)'),
    (r'entropy_reg\b',         '# Entropy regularization: encourages policy exploration'),
    (r'iqn|quantile|cvar',     '# IQN/CVaR: distributional RL with tail-risk control'),
    (r'regime\b',              '# Market regime: trending / ranging / volatile'),
    (r'kill.switch\b',         '# Kill switch: emergency halt — closes all positions immediately'),
    # General Python
    (r'super\(\)\.__init__\(', '# Calls the parent class constructor'),
    (r'print\(',               '# Prints output to stdout'),
    (r'logger\.',              '# Logs a message via the module logger'),
    (r'os\.makedirs\(',        '# Creates directory (and parents) if they do not exist'),
    (r'os\.path\.exists\(',    '# Returns True if the path exists on disk'),
    (r'os\.path\.join\(',      '# Joins path components into a single path string'),
    (r'\.append\(',            '# Appends an item to the end of the list'),
    (r'\.extend\(',            '# Extends list by appending all items from iterable'),
    (r'enumerate\(',           '# Iterates with both index and value'),
    (r'zip\(',                 '# Iterates over multiple iterables in parallel'),
    (r'isinstance\(',          '# Checks if object is an instance of given type(s)'),
    (r'len\(',                 '# Returns the number of items'),
    (r'range\(',               '# Generates a sequence of integers'),
    (r'setattr\(',             '# Sets a named attribute on an object dynamically'),
    (r'getattr\(',             '# Gets a named attribute from an object dynamically'),
    (r'hasattr\(',             '# Returns True if object has the named attribute'),
    (r'sorted\(',              '# Returns a new sorted list'),
    (r'parser\.add_argument\(','# Registers a CLI argument for argparse'),
    (r'args\s*=\s*parser\.parse_args\(','# Parses CLI arguments into an args namespace'),
    (r'logging\.getLogger',    '# Gets (or creates) a named logger for this module'),
    (r'\.train\(\)',           '# Switches model to training mode (enables Dropout, BN)'),
    (r'\.eval\(\)',            '# Switches model to evaluation mode (disables Dropout)'),
    (r'checkpoint|ckpt',       '# Checkpoint: saved model state for resuming training'),
]


def get_import_comment_en(code: str):
    m = re.match(r'^from\s+([\w.]+)\s+import\s+(.+)', code)
    if m:
        mod = m.group(1).split('.')[-1].lower()
        names = m.group(2).strip()
        if len(names) > 40:
            names = names[:40] + '...'
        desc = IMPORT_MAP_EN.get(mod, f'{m.group(1)} module')
        return f'# Import {names} from {desc}'
    m = re.match(r'^import\s+([\w.]+)\s+as\s+(\w+)', code)
    if m:
        desc = IMPORT_MAP_EN.get(m.group(2).lower(),
               IMPORT_MAP_EN.get(m.group(1).split('.')[-1].lower(),
               f'{m.group(1)} library'))
        return f'# Import {desc} as "{m.group(2)}"'
    m = re.match(r'^import\s+([\w.]+)', code)
    if m:
        key = m.group(1).split('.')[-1].lower()
        desc = IMPORT_MAP_EN.get(key, f'{m.group(1)} library')
        return f'# Import {desc}'
    return None


def get_def_comment_en(code: str):
    m = re.match(r'^class\s+(\w+)', code)
    if m:
        return f'# ── Class [{m.group(1)}]: groups related data and behaviour ──'
    m = re.match(r'^def\s+(\w+)\s*\(', code)
    if m:
        name = m.group(1)
        special = {
            '__init__':      'Constructor — runs when the object is created',
            '__repr__':      'Developer-readable string representation',
            '__str__':       'Human-readable string representation',
            '__len__':       'Returns the length of the object',
            '__call__':      'Makes the object callable like a function',
            'forward':       'Defines the forward pass of the neural network',
            'train_step':    'Executes one training step (forward + backward + update)',
            'select_action': 'Selects an action (HOLD / LONG / SHORT) given the current state',
            'save_checkpoint': 'Saves current model weights to disk',
            'load_checkpoint': 'Loads model weights from a checkpoint file',
            'fit':           'Fits the model to training data',
            'predict':       'Runs inference and returns predictions',
            'reset':         'Resets all internal state to initial values',
            'update':        'Updates internal state or parameters',
        }
        if name in special:
            return f'# [{name}] {special[name]}'
        if name.startswith('__'):
            return f'# [{name}] Special / dunder method'
        if name.startswith('_'):
            return f'# [{name}] Private helper function'
        return f'# [{name}] Function definition'
    return None


def get_control_comment_en(code: str):
    if re.match(r'^if\s+not\s+', code):   return '# Branch: executes only when condition is False'
    if re.match(r'^if\s+',       code):   return '# Branch: executes only when condition is True'
    if re.match(r'^elif\s+',     code):   return '# Branch: previous condition was False, try this one'
    if re.match(r'^else\s*:',    code):   return '# Branch: all previous conditions were False'
    if re.match(r'^for\s+\w+\s+in\s+range\(', code): return '# Loop: iterate a fixed number of times'
    if re.match(r'^for\s+',      code):   return '# Loop: iterate over each item in the sequence'
    if re.match(r'^while\s+True\s*:', code): return '# Loop: run indefinitely until explicitly broken'
    if re.match(r'^while\s+',    code):   return '# Loop: repeat while condition holds'
    if re.match(r'^return\b',    code):   return '# Returns a value to the caller'
    if re.match(r'^raise\s+',    code):   return '# Raises an exception to signal an error'
    if re.match(r'^try\s*:',     code):   return '# Try block: attempt code that might raise an exception'
    if re.match(r'^except\s*',   code):   return '# Except block: handles a raised exception'
    if re.match(r'^finally\s*:', code):   return '# Finally block: always executes (cleanup)'
    if re.match(r'^with\s+torch\.no_grad\(\)', code):
        return '# Context: disable gradient tracking for inference (saves memory)'
    if re.match(r'^with\s+',     code):   return '# Context manager: resource is opened and auto-closed'
    if re.match(r'^pass\s*$',    code):   return '# No-op placeholder'
    if re.match(r'^break\s*$',   code):   return '# Exit the enclosing loop immediately'
    if re.match(r'^continue\s*$',code):   return '# Skip the rest of this iteration'
    if re.match(r'^assert\s+',   code):   return '# Assertion: raises AssertionError if condition is False'
    if re.match(r'^yield\s+',    code):   return '# Generator: yields one value to the caller'
    return None


def get_decorator_comment_en(code: str):
    if re.match(r'^@dataclass',    code): return '# Decorator: auto-generate __init__, __repr__, etc.'
    if re.match(r'^@property',     code): return '# Decorator: expose method as a read-only attribute'
    if re.match(r'^@staticmethod', code): return '# Decorator: static method — no self or cls'
    if re.match(r'^@classmethod',  code): return '# Decorator: class method — receives cls as first arg'
    if re.match(r'^@torch\.jit',   code): return '# Decorator: compile function with TorchScript'
    if re.match(r'^@',             code): return '# Decorator: modifies the function / class below'
    return None


def get_assignment_comment_en(code: str):
    if re.search(r'self\.\w+\s*=\s*nn\.Parameter\(', code):
        return '# Registers tensor as a learnable parameter (tracked by autograd)'
    if re.search(r'self\.\w+\s*=\s*nn\.\w+\(', code):
        return '# Stores a neural network layer as an attribute of this module'
    if re.match(r'^device\s*=', code):
        return '# Target compute device: CUDA GPU or CPU'
    if re.match(r'^lr\s*=', code):
        return '# Learning rate: step size for each parameter update'
    if re.match(r'^gamma\s*=', code):
        return '# Discount factor γ ∈ (0,1]: weight for future rewards'
    if re.match(r'^batch.size\s*=', code):
        return '# Number of samples per gradient update step'
    if re.match(r'^(n_)?epoch\w*\s*=', code):
        return '# Number of full passes over the training dataset'
    return None


def generate_comment_en(code: str):
    if not code:
        return None
    stripped = code.lstrip()
    if stripped.startswith('#') or '"""' in code or "'''" in code:
        return None
    if re.match(r'^[\)\]\},]+,?\s*$', stripped):
        return None

    for getter in [get_import_comment_en, get_decorator_comment_en,
                   get_def_comment_en, get_control_comment_en,
                   get_assignment_comment_en]:
        c = getter(stripped)
        if c:
            return c

    for pattern, comment in PATTERN_COMMENTS_EN:
        if re.search(pattern, stripped, re.IGNORECASE):
            return comment

    if 'super().__init__' in stripped:
        return '# Call parent class __init__'
    if re.search(r'\.train\(\)', stripped):
        return '# Set model to training mode'
    if re.search(r'\.eval\(\)', stripped):
        return '# Set model to evaluation mode'
    if re.search(r'checkpoint|ckpt', stripped, re.IGNORECASE):
        return '# Checkpoint: save/load model state'
    return None


def process_file(filepath: str) -> None:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='cp949') as f:
            lines = f.readlines()

    result = []
    in_multiline = False
    multiline_char = ''
    added = 0

    for line in lines:
        raw = line.rstrip('\n')
        stripped = raw.lstrip()
        indent_n = len(raw) - len(stripped)
        indent_str = ' ' * indent_n

        # Track multiline strings
        if not in_multiline:
            dq = stripped.count('"""')
            sq = stripped.count("'''")
            if dq % 2 == 1:
                in_multiline = True; multiline_char = '"""'
                result.append(line); continue
            if sq % 2 == 1:
                in_multiline = True; multiline_char = "'''"
                result.append(line); continue
        else:
            if multiline_char in stripped:
                in_multiline = False
            result.append(line); continue

        # Skip blank lines
        if not stripped:
            result.append(line); continue

        # ── Case 1: line already has an English inline comment — keep it ──
        has_english_inline = bool(re.search(r'  #\s*[A-Za-z\[★]', raw))
        if has_english_inline:
            result.append(line); continue

        # ── Case 2: line has a Korean inline comment — append English ──
        has_korean_inline = bool(re.search(r'  #[^#\n]*[가-힣]', raw))
        if has_korean_inline:
            # Extract code part
            code_part = re.split(r'\s{2,}#', raw, maxsplit=1)[0].rstrip()
            en = generate_comment_en(code_part)
            if en:
                # Append English after Korean
                result.append(raw.rstrip() + '  ' + en + '\n')
                added += 1
            else:
                result.append(line)
            continue

        # ── Case 3: line starts with Korean comment line — leave as-is ──
        if re.match(r'\s*#[^#\n]*[가-힣]', raw):
            result.append(line); continue

        # ── Case 4: pure code line (no comment at all) — add English ──
        if stripped.startswith('#'):
            result.append(line); continue
        if '"""' in stripped or "'''" in stripped:
            result.append(line); continue
        if re.match(r'^[\)\]\},]+,?\s*$', stripped):
            result.append(line); continue

        en = generate_comment_en(raw)
        if en:
            if len(raw) + len(en) + 2 <= 120:
                result.append(raw + '  ' + en + '\n')
            else:
                result.append(indent_str + en + '\n')
                result.append(line)
            added += 1
        else:
            result.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(result)

    print(f'[done] {filepath}  ({len(lines)} → {len(result)} lines, +{added} EN comments)')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python _auto_comment_en.py <file1.py> [file2.py ...]')
        sys.exit(1)
    for path in sys.argv[1:]:
        process_file(path)
