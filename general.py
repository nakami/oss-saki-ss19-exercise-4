from framework.company import Company
from experts.obscure_expert import ObscureExpert

Company.A
my_obscure_expert = ObscureExpert(Company.A)

print(my_obscure_expert)
#print(dir(my_obscure_expert))
print(type(my_obscure_expert._ObscureExpert__answers))

keys_in_dict = list(my_obscure_expert._ObscureExpert__answers.keys())

print(f'keys_in_dict:\t\t{keys_in_dict}')

amount_days_a = len(my_obscure_expert._ObscureExpert__answers[Company.A])
amount_days_b = len(my_obscure_expert._ObscureExpert__answers[Company.B])

print(f'amount_days_a:\t\t{amount_days_a}')
print(f'amount_days_b:\t\t{amount_days_b}')

items_a = list(my_obscure_expert._ObscureExpert__answers[Company.A].items())

print(f'first item in A:\t{items_a[0]}')
print(f'last item in A:\t\t{items_a[-1]}')