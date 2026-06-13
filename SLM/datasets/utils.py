import torch

def dec2bin(n, bits=None):
    """
    Convert integers to binary.
    
    Args:
            n: The numbers to convert (tensor of size [*]).
         bits: The length of the representation.
    Returns:
        A tensor (size [*, bits]) with the binary representations.
    """
    if bits is None:
        bits = (x.max() + 1).log2().ceil().item()
    x = x.int()
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def dec2base(n, b, length=None):
    """
    Convert integers into a different base.
    
    Args:
            n: The numbers to convert (tensor of size [*]).
            b: The base (integer).
       length: The length of the representation.
    Returns:
        A tensor (size [*, length]) containing the input numbers in the new base.
    """
    digits = []
    while n.sum():
        digits.append(n % b)
        n = n.div(b, rounding_mode='floor')
    if length:
        assert len(digits) <= length, "Length required is too small to represent input numbers!"
        digits += [torch.zeros(len(n), dtype=int)] * (length - len(digits))
    return torch.stack(digits[::-1]).t()

def base2dec(t, b):
    """
    Convert tuples of s integers in base b into integers in base b**s.
    
    Args:
            t: tuples to convert (tesor of size [*,s]).
            b: the base (integer).
    
    Returns:
        A tensor (size [*]) with the inputs in the new base
    """
    length = t.size(-1)  # Length of the tuples, which gives the number of digits    
    # Create powers of b: [b**(length-1), b**(length-2), ..., b**0]
    powers = torch.tensor([b ** i for i in reversed(range(length))], dtype=t.dtype, device=t.device)
    # Multiply the tensor by the powers of b and sum along the last dimension
    result = torch.sum(t * powers, dim=-1)
    
    return result