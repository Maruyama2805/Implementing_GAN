Implementação de GAN (Generative Adversarial Network) com PyTorch
Este repositório contém uma implementação de uma Rede Adversarial Geradora (GAN), especificamente uma DCGAN (Deep Convolutional GAN), treinada no dataset MNIST. O código está disponível no notebook GAN.ipynb.

Este projeto serve como uma introdução prática à arquitetura de GANs, que é a base para técnicas mais avançadas como a AnoGAN, amplamente utilizada para detecção de anomalias em diversos campos, incluindo a engenharia de controle e automação.

1. Conceitos-Chave
De acordo com os materiais de referência fornecidos, abaixo estão os conceitos fundamentais.

1.1. O que é uma GAN (Generative Adversarial Network)?
Uma Rede Adversarial Geradora (GAN) é um tipo de arquitetura de machine learning que consiste em duas redes neurais competindo entre si. Esse "jogo" permite que o sistema crie dados sintéticos novos e realistas. As duas redes são:

O Gerador (Generator): Sua função é criar dados "falsos". Ele recebe um vetor de ruído aleatório (chamado de espaço latente, z) e tenta transformá-lo em uma saída que se pareça com os dados reais (por exemplo, uma imagem de um dígito escrito à mão).

O Discriminador (Discriminator): Sua função é atuar como um "crítico" ou "policial". Ele recebe tanto dados reais (do conjunto de treinamento) quanto dados falsos (criados pelo Gerador) e tenta classificar corretamente se o dado é real ou falso.

Durante o treinamento, o Gerador fica progressivamente melhor em "enganar" o Discriminador, enquanto o Discriminador fica melhor em "identificar" as fraudes. Esse processo de "soma zero" (minimax) termina quando o Gerador se torna tão eficaz que suas saídas falsas são quase indistinguíveis das reais.

1.2. O que é AnoGAN?
AnoGAN (e suas variações como a f-AnoGAN) é uma arquitetura que aplica o conceito de GANs para a detecção de anomalias não supervisionada.

A principal ideia é treinar uma GAN (Gerador e Discriminador) exclusivamente com dados normais (ou "saudáveis"). Após o treinamento, o Gerador aprendeu a "distribuição" ou as características fundamentais dos dados normais.

Quando um novo dado (uma nova imagem, por exemplo) precisa ser testado, o processo é o seguinte:

A nova imagem é apresentada ao sistema.

O sistema tenta encontrar o vetor de ruído (z) no espaço latente que, quando passado pelo Gerador treinado, produz a imagem mais parecida possível com a imagem de teste.

Calcula-se uma pontuação de anomalia (anomaly score), que é baseada no "erro de reconstrução" (a diferença entre a imagem de teste e a imagem gerada).

Se a imagem de teste for normal, o Gerador (que só viu dados normais) conseguirá recriá-la com facilidade, resultando em um erro baixo. Se a imagem de teste for uma anomalia (algo que o Gerador nunca viu durante o treino), ele falhará em recriá-la fielmente, resultando em um erro alto. Esse erro alto sinaliza que o dado é uma anomalia.

2. Importância para Engenharia de Controle e Automação
As arquiteturas GAN e AnoGAN têm um impacto significativo na engenharia de controle e automação, principalmente para garantir segurança, qualidade e eficiência em sistemas industriais.

Manutenção Preditiva e Detecção de Falhas: Em um sistema de automação (como uma linha de montagem ou uma turbina), sensores monitoram vibração, temperatura, pressão e som. Uma AnoGAN pode ser treinada usando dados desses sensores durante a operação normal (saudável) da máquina. O sistema pode, então, monitorar a máquina em tempo real. Qualquer desvio nos dados dos sensores (que gere um alto "erro de reconstrução" na AnoGAN) é classificado como uma anomalia. Isso permite que engenheiros de controle identifiquem uma falha iminente (ex: um rolamento desgastado) antes que ela cause uma parada catastrófica, habilitando a manutenção preditiva.

Controle de Qualidade Automatizado (Visão Computacional): Em linhas de produção automatizadas, câmeras inspecionam produtos (ex: placas de circuito, soldas, tecidos). Treinar um sistema de visão para encontrar todos os defeitos possíveis é difícil, pois os defeitos podem ser raros ou imprevisíveis. Com a AnoGAN, o sistema é treinado apenas com imagens de produtos perfeitos. Qualquer produto que apresente um defeito visual (um arranhão, um componente desalinhado) será marcado como uma anomalia, sendo automaticamente rejeitado pelo sistema de controle.

Simulação e Geração de Dados (Digital Twins): Controladores robustos precisam ser testados em cenários de falha, que muitas vezes são raros e perigosos de se replicar em equipamentos reais. As GANs (o Gerador) podem ser usadas para gerar dados sintéticos realistas que simulam essas falhas raras (ex: picos de sensor, falhas de atuador). Esses dados sintéticos podem ser usados para treinar controladores mais seguros ou para validar "Gêmeos Digitais" (Digital Twins) do processo industrial.

3. Sobre esta Implementação (GAN.ipynb)
Este notebook foca no primeiro passo: construir a GAN base. Ele não é uma AnoGAN, mas implementa o Gerador e o Discriminador que são os componentes centrais de uma AnoGAN.

Framework: PyTorch

Dataset: MNIST (dígitos escritos à mão, normalizados para [-1, 1])

Arquitetura:

Generator: Usa camadas ConvTranspose2d (transposição convolucional) para fazer o upsampling do vetor latente (z) de 100 dimensões até uma imagem 1x28x28.

Discriminator: Usa camadas Conv2d (convolucional) para fazer o downsampling de uma imagem 1x28x28 até uma única previsão (Real/Falsa).

Treinamento: O notebook itera por 50 épocas, treinando alternadamente o Discriminador e o Gerador. O progresso é visualizado usando tqdm.

Resultado: O modelo treinado do Gerador é salvo em generator_mnist.pth. A seção final do notebook carrega este arquivo e gera 16 imagens sintéticas de dígitos.

4. Como Usar
Pré-requisitos
Você precisará das seguintes bibliotecas Python:

torch

torchvision

matplotlib

tqdm

Executando o Projeto
Treinamento:

Abra o GAN.ipynb em um ambiente Jupyter (como Google Colab ou localmente).

Execute as células sequencialmente. O script irá baixar o dataset MNIST automaticamente.

Ao final do treinamento (cerca de 30 minutos em uma GPU T4), o modelo generator_mnist.pth será salvo.

Geração de Imagens:

Execute as células na seção "Visualizando resultados".

Este script irá carregar o generator_mnist.pth, gerar 16 imagens e exibi-las usando matplotlib. Uma cópia da grade de imagens também será salva como generated_mnist_grid.png.

5. Referências
Guias de Machine Learning | Google Developers (PT-BR)

f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks (TUMdlma)
