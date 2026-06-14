# Plano de Projeto — PPE Detector
> Detecção de EPIs com OpenCV + YOLOv5

---

## Ideia Central

Construir um sistema que detecta, em imagens e vídeos, se trabalhadores estão usando EPIs corretos. A pipeline combina **OpenCV** para pré-processamento de imagem e **YOLOv5** para detecção dos equipamentos, emitindo alertas visuais quando houver violação.

O pré-processamento com OpenCV é justificado pela ausência de augmentação interna robusta no YOLOv5, tornando a qualidade da imagem de entrada relevante para o desempenho do modelo — especialmente em condições variáveis de iluminação comuns em obras.

O projeto é dividido em duas etapas:
1. **Validação com imagem estática** — garante que a pipeline funciona antes de lidar com vídeo
2. **Detecção em vídeo/webcam** — aplica tudo em tempo real

---

## Referência Acadêmica

O projeto é baseado no artigo **Nath et al. (2020) — "Deep learning for site safety: Real-time detection of personal protective equipment"**, publicado na *Automation in Construction*, que utiliza YOLOv3 como arquitetura base. Adotamos YOLOv5 por ser a evolução direta, mantendo compatibilidade com as decisões do artigo e facilitando a execução local.

---

## Dataset

Dataset disponível no Kaggle com 8 classes de EPI (presença e ausência):

```python
import kagglehub
path = kagglehub.dataset_download("anuragraj03/ppe-detection-m")
print("Path to dataset files:", path)
```

**Classes:**

| Id | Classe | Significado |
|----|--------|-------------|
| 0  | no-safety-glove  | sem luvas |
| 1  | no-safety-helmet | sem capacete |
| 2  | no-safety-shoes  | sem calçado de segurança |
| 3  | no-welding-glass | sem óculos de solda |
| 4  | safety-glove     | com luvas |
| 5  | safety-helmet    | com capacete |
| 6  | safety-shoes     | com calçado de segurança |
| 7  | welding-glass    | com óculos de solda |

---

## Por que YOLOv5?

- **Leve:** roda bem em CPU, sem necessidade de GPU dedicada
- **Simples de configurar:** repositório oficial com documentação clara
- **Justifica pré-processamento:** menos augmentação interna que v8, tornando o OpenCV relevante
- **Próximo do artigo de referência:** Nath et al. usaram v3; v5 é a evolução direta
- **Comparação possível:** dá para comparar resultados com e sem pré-processamento OpenCV

---

## Estrutura de Arquivos

```
ppe_detector/
│
├── config.py              # configurações gerais (paths, thresholds, cores)
├── preprocessor.py        # pipeline de pré-processamento com OpenCV
├── detector.py            # inferência YOLOv5
├── alerter.py             # lógica de análise e alerta
├── visualizer.py          # desenho de resultados na imagem
│
├── run_image.py           # execução com imagem estática (fazer primeiro)
├── run_video.py           # execução com vídeo / webcam (fazer depois)
│
├── train/
│   └── train.py           # script de treino do modelo
│
├── models/
│   └── best.pt            # modelo treinado (gerado pelo treino)
│
├── test_images/           # imagens para testar a pipeline
└── dataset/               # dataset baixado do Kaggle
    └── dataset.yaml
```

---

## Responsabilidade de cada arquivo

| Arquivo | Responsabilidade |
|---------|-----------------|
| `config.py` | Centraliza paths, thresholds e parâmetros — alterar aqui reflete em tudo |
| `preprocessor.py` | Trata o frame com OpenCV antes de enviar ao YOLOv5 |
| `detector.py` | Roda a inferência do YOLOv5 e devolve detecções padronizadas |
| `alerter.py` | Analisa as detecções e separa violações de situações seguras |
| `visualizer.py` | Desenha bounding boxes, alertas e informações na imagem |
| `run_image.py` | Orquestra a pipeline para uma imagem estática |
| `run_video.py` | Orquestra a pipeline para vídeo frame a frame |
| `train/train.py` | Treina o YOLOv5 com o dataset de EPIs |

---

## Fluxo da Pipeline

```
Frame bruto
    │
    ▼
[preprocessor.py]
    └── tratamento de imagem com OpenCV (a definir)
    │
    ▼
[detector.py]
    └── YOLOv5 → detecta e classifica EPIs
    │
    ▼
[alerter.py]
    └── separa detecções seguras / violações
    │
    ▼
[visualizer.py]
    ├── bounding boxes verdes (com EPI) / vermelhos (sem EPI)
    └── alerta na tela quando houver violação
```

---

## Ordem de execução

```
1. train/train.py      → treina o modelo com o dataset
2. run_image.py        → valida a pipeline com imagem estática
3. run_video.py        → testa em tempo real com vídeo ou webcam
```

---

## Cronograma — 15 dias

### Semana 1 — Treino

| Dia | Tarefa |
|-----|--------|
| 1–2 | Configurar ambiente, baixar dataset, instalar YOLOv5 |
| 3–5 | Escrever e rodar `train/train.py` |
| 6–7 | Validar modelo treinado (checar mAP por classe) |

### Semana 2 — Pipeline

| Dia | Tarefa |
|-----|--------|
| 8   | `config.py` + definir e implementar `preprocessor.py` |
| 9   | `detector.py` |
| 10  | `alerter.py` + `visualizer.py` |
| 11  | `run_image.py` + testes com imagens estáticas |
| 12  | `run_video.py` + testes com vídeo |
| 13–15 | Ajustes, testes finais, documentação e apresentação |

---

## Conceitos de Visão Computacional envolvidos

| Conceito | Onde será aplicado |
|----------|--------------------|
| Pré-processamento de imagem | `preprocessor.py` com OpenCV (a definir) |
| Transfer learning | YOLOv5 pré-treinado no COCO, fine-tuning no dataset de EPIs |
| Detecção de objetos | YOLOv5 com bounding boxes |
| Non-Maximum Suppression | Interno ao YOLOv5 |
| mAP (mean Average Precision) | Métrica de avaliação do modelo |
| IoU (Intersection over Union) | Critério de avaliação das bounding boxes |
