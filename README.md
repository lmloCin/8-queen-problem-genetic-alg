# 8-queen-problem-genetic-alg

## Primeira parte:
- Representação (genótipo): Permutação de string de bits
- Recombinação: “cut-and-crossfill” crossover
- Probabilidade de Recombinação: 90%
- Mutação: troca de genes
- Probabilidade de Mutação: 40%
- Seleção de pais: ranking - Melhor de 2 de 5 escolhidos
aleatoriamente
- Seleção de sobreviventes: substituição do pior
- Tamanho da população: 100
- Número de filhos gerados: 2
- Inicialização: aleatória
- Condição de término: Encontrar a solução, ou 10.000
avaliações de fitness
- Fitness? 

## Segunda parte:
- Implementar possíveis melhorias mudando:
- Representação
- Recombinação
- Mutação
- Seleção de pais - roleta?
- Seleção de sobreviventes: geracional ou substituição
do pior
- Tamanho da população: 10? 30? 50? 70? 120? 200?
- O fitness pode ser melhorado?

## Avaliação do Projeto

O objetivo é avaliar se as modificações propostas para a solução
do problema das 8 rainhas foram eficientes e eficazes e porque
essas alterações levaram a melhora/piora.
Para cada implementação devem ser feitas 30 execuções e
analisar
- Em quantas execuções o algoritmo convergiu (n o /30
execuções);
- Em que iteração o algoritmo convergiu (média e desvio
padrão);
- Número de indivíduos que convergiram por execução;
- Fitness médio da população em cada uma das 30
execuções;
- Colocar gráficos de convergência com a média e o melhor
indivíduo por iteração;
- Fitness médio alcançado nas 30 execuções (média e desvio
padrão);
- Análise adicional: Quantas iterações são necessárias para
toda a população convergir?


## Guia de arquivos

- firsttrysolve.py --> Primeira tentativa de resolver o problema (não seguia 100% os requisitos do projeto)
- firsthalf.py --> Solução da primeira parte do projeto
- secondhalf.py --> Solução para a segunda parte do projeto (em andamento)
- inteligent_c_m.py --> Resolução tentando conceitos de mutação e geração de pais de forma mais inteligente
- secondhalf.py --> Resolução utilizando torneio binário e seleção geracional
- secondhalf_inteligent.py --> Resolução juntando as melhorias de secondhalf.py e inteligent_c_m.py
- secondhalf_n_inteligent.py --> Uso da resolução de secondhalf_inteligent.py aplicado a N rainhas em um tabuleiro NxN
