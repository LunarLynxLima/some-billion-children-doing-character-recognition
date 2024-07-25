## Custome maths operation used by the model (linear)
import torch
import numpy as np

global default_opt
default_opt = False

class MathUtils:
    @staticmethod
    def exposed_calulation(a, b, operation = "*", q = "What is",user_calc_override=False, auto_compute = True):
        correct =  MathUtils.autocompute(a,b,operation)
        if auto_compute: return correct
        
        q = f"{q} {a}{operation}{b}? " 
        ans = (input(f"{q}"))
        try: ans = float(ans)
        except: 
            if ans == "?": 
                print(f"Don't know such complex calculation? Let us help you simplify it.",
                      f"To be implemented in the future.")
                return MathUtils.simplify(a,b,operation)
            else:
                print(f"Invalid input, Program fall backs to Autocalulation.\n",
                    f"The correct answer of {q} is {correct}.")
                return correct
        
        if user_calc_override: return ans
        if float(ans) == correct: return ans
        else: 
            print(f"Your answer ({(int(ans))}) is incorrect. The correct answer of '{q}' is {correct}.")
            return correct
    
    #! TODO: Implement the simplify method (Cuurently it call the auto-compute method)
    @staticmethod
    def simplify(a, b, operation):
        return exposed_calulation.autocompute(a,b,operation)
    @staticmethod
    def autocompute(a, b, operation):
        if operation == "*": return a*b
        elif operation == "+": return a+b
        elif operation == "-": return a-b
        elif operation == "/": return a/b
        else: raise ValueError("Invalid operation. Please use one of the following: '+', '-', '*', '/'")
    
class CustomNumber:
    def __init__(self, value):
        self.value = value

    def __mul__(self, other):
        if not isinstance(other, CustomNumber):
            return MathUtils.exposed_calulation(self.value, other, operation = "*")
        else:
            return MathUtils.exposed_calulation(self.value, other.value, operation = "*")
    
    def __sub__(self, other):
        if not isinstance(other, CustomNumber):
            return MathUtils.exposed_calulation(self.value, other, operation = "-")
        else:
            return MathUtils.exposed_calulation(self.value, other.value, operation = "-")
    
    def __add__(self, other):
        if not isinstance(other, CustomNumber):
            return MathUtils.exposed_calulation(self.value, other, operation = "+")
        else:
            return MathUtils.exposed_calulation(self.value, other.value, operation = "+")
    
    def __truediv__(self, other):
        if not isinstance(other, CustomNumber):
            return MathUtils.exposed_calulation(self.value, other, operation = "/")
        else:
            return MathUtils.exposed_calulation(self.value, other.value, operation = "/")
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __rtruediv__(self, other):
        return self.__truediv__(other)
    
    def __eq__(self, value: object) -> bool:
        return self.value == value


    def __repr__(self):
        return f"{self.value}"
class CustomArray(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        # Perform the element-wise conversion to custom Number
        obj = cls._convert_to_custom_numbers_array(obj)
        return obj

    @staticmethod
    def _convert_to_custom_numbers_array(array):
        def convert_recursive(element):
            if isinstance(element, np.ndarray) and element.ndim > 0:
                return np.array([convert_recursive(x) for x in element])
            else:
                return CustomNumber(element)
        # Check if the array is a scalar
        if np.isscalar(array) or array.ndim == 0:
            return CustomNumber(array)
        return convert_recursive(array)

    @staticmethod
    def _convert_to_np_array(array):
        def convert_recursive(element):
            if isinstance(element, CustomNumber):
                return element.value
            elif isinstance(element, np.ndarray):
                return np.array([convert_recursive(x) for x in element])
            else:
                return element
        return convert_recursive(array)

    @staticmethod
    def _convert_to_torch_tensor(array):
        return torch.tensor(CustomArray._convert_to_np_array(array))
    
    # transpose custom array with custom numbers
    @staticmethod
    def _transpose(array):
        array = CustomArray._convert_to_torch_tensor(array)
        array = array.T
        return CustomArray._convert_to_custom_numbers_array(array)
    
    # def __mul__(self, other):
    #     if not isinstance(other, CustomArray):
    #         raise ValueError("Multiplication is only supported between CustomArray instances.")
    #     return np.array([a * b for a, b in zip(self.ravel(), other.ravel())]).reshape(self.shape).view(CustomArray)
    
    # def __sub__(self, other):
    #     if not isinstance(other, CustomArray):
    #         raise ValueError("Subtraction is only supported between CustomArray instances.")
    #     return np.array([a - b for a, b in zip(self.ravel(), other.ravel())]).reshape(self.shape).view(CustomArray)
    
    # def __add__(self, other):
    #     if not isinstance(other, CustomArray):
    #         raise ValueError("Addition is only supported between CustomArray instances.")
    #     return np.array([a + b for a, b in zip(self.ravel(), other.ravel())]).reshape(self.shape).view(CustomArray)
    
    # def __truediv__(self, other):
    #     if not isinstance(other, CustomArray):
    #         raise ValueError("Division is only supported between CustomArray instances.")
    #     return np.array([a / b for a, b in zip(self.ravel(), other.ravel())]).reshape(self.shape).view(CustomArray)

    def __repr__(self):
        return f"CustomArray({super().__repr__()})"


def main():
    # Create nd arrays and convert to CustomNumber instances
    arr_a = [[1, 2], [1,2 ]]
    arr_b = [[1, 2], [1,2 ]]
    arr_a = torch.tensor(arr_a, requires_grad=False)
    arr_b = torch.tensor(arr_b, requires_grad=False)
    arr_a = CustomArray(arr_a)
    arr_b = CustomArray(arr_b)

    # Perform element-wise multiplication using the custom logic
    result_arr = arr_a @ arr_b

    print("Custom Multiply Array Result:\n", result_arr)

if __name__ == "__main__": main()





# Arithmetic Operators:
# __add__(self, other) for addition (+)
# __sub__(self, other) for subtraction (-)
# __mul__(self, other) for multiplication (*)
# __truediv__(self, other) for true division (/)
# __floordiv__(self, other) for floor division (//)
# __mod__(self, other) for modulus (%)
# __pow__(self, other) for exponentiation (**)

# Comparison Operators:
# __eq__(self, other) for equality (==)
# __ne__(self, other) for inequality (!=)
# __lt__(self, other) for less than (<)
# __le__(self, other) for less than or equal to (<=)
# __gt__(self, other) for greater than (>)
# __ge__(self, other) for greater than or equal to (>=)

# Unary Operators:
# __neg__(self) for negation (-)
# __pos__(self) for unary positive (+)
# __abs__(self) for absolute value (abs())