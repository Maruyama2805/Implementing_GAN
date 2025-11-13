# Implementação de GAN (Generative Adversarial Network) com PyTorch

Este repositório contém uma implementação de uma **Rede Adversarial Geradora (GAN)**, especificamente uma DCGAN (Deep Convolutional GAN), treinada no dataset MNIST. O código está disponível no notebook `GAN.ipynb`.

Este projeto serve como uma introdução prática à arquitetura de GANs, que é a base para técnicas mais avançadas como a **AnoGAN**, amplamente utilizada para detecção de anomalias em diversos campos, incluindo a engenharia de controle e automação.

## 1. Conceitos-Chave

De acordo com os materiais de referência fornecidos, abaixo estão os conceitos fundamentais.

### 1.1. O que é uma GAN (Generative Adversarial Network)?

Uma Rede Adversarial Geradora (GAN) é um tipo de arquitetura de machine learning que consiste em duas redes neurais competindo entre si. Esse "jogo" permite que o sistema crie dados sintéticos novos e realistas. As duas redes são:

* **O Gerador (Generator):** Sua função é criar dados "falsos". Ele recebe um vetor de ruído aleatório (chamado de espaço latente, *z*) e tenta transformá-lo em uma saída que se pareça com os dados reais (por exemplo, uma imagem de um dígito escrito à mão).
* **O Discriminador (Discriminator):** Sua função é atuar como um "crítico" ou "policial". Ele recebe tanto dados reais (do conjunto de treinamento) quanto dados falsos (criados pelo Gerador) e tenta classificar corretamente se o dado é real ou falso.

Durante o treinamento, o Gerador fica progressivamente melhor em "enganar" o Discriminador, enquanto o Discriminador fica melhor em "identificar" as fraudes. Esse processo de "soma zero" (minimax) termina quando o Gerador se torna tão eficaz que suas saídas falsas são quase indistinguíveis das reais.

### 1.2. O que é AnoGAN?

**AnoGAN** (e suas variações como a f-AnoGAN) é uma arquitetura que aplica o conceito de GANs para a **detecção de anomalias não supervisionada**.

A principal ideia é treinar uma GAN (Gerador e Discriminador) exclusivamente com dados **normais** (ou "saudáveis"). Após o treinamento, o Gerador aprendeu a "distribuição" ou as características fundamentais dos dados normais.

Quando um novo dado (uma nova imagem, por exemplo) precisa ser testado, o processo é o seguinte:

1.  A nova imagem é apresentada ao sistema.
2.  O sistema tenta encontrar o vetor de ruído (`z`) no espaço latente que, quando passado pelo **Gerador treinado**, produz a imagem mais parecida possível com a imagem de teste.
3.  Calcula-se uma **pontuação de anomalia (anomaly score)**, que é baseada no "erro de reconstrução" (a diferença entre a imagem de teste e a imagem gerada).

Se a imagem de teste for **normal**, o Gerador (que só viu dados normais) conseguirá recriá-la com facilidade, resultando em um **erro baixo**. Se a imagem de teste for uma **anomalia** (algo que o Gerador nunca viu durante o treino), ele falhará em recriá-la fielmente, resultando em um **erro alto**. Esse erro alto sinaliza que o dado é uma anomalia.

## 2. Importância para Engenharia de Controle e Automação

As arquiteturas GAN e AnoGAN têm um impacto significativo na engenharia de controle e automação, principalmente para garantir segurança, qualidade e eficiência em sistemas
