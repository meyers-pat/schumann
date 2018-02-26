import schumann
print schumann.__path__
ST = open(schumann.__path__[0] + '/templates/params_template.txt','r').read()
