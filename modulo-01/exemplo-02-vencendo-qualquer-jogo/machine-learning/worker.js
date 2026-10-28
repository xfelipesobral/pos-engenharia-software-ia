importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

const MODEL_PATH = `yolov5n_web_model/model.json`;
const LABELS_PATH = `yolov5n_web_model/labels.json`;
const INPUT_MODEL_DIMENTIONS = 640
const CLASS_THRESHOLD = 0.4

let _labels = []
let _model = null
async function loadModelAndLabels() {
    await tf.ready()

    _labels = await (await fetch(LABELS_PATH)).json()
    _model = await tf.loadGraphModel(MODEL_PATH)

    // warmup
    const dummyInput = tf.ones(_model.inputs[0].shape)
    await _model.executeAsync(dummyInput)
    tf.dispose(dummyInput)

    postMessage({ type: 'model-loaded' })

}

/**
 * Pré-processa a imagem para o formato aceito pelo YOLO:
 * - tf.browser.fromPixels(): converte ImageBitmap/ImageData para tensor [H, W, 3]
 * - tf.image.resizeBilinear(): redimensiona para [INPUT_DIM, INPUT_DIM]
 * - .div(255): normaliza os valores para [0, 1]
 * - .expandDims(0): adiciona dimensão batch [1, H, W, 3]
 *
 * Uso de tf.tidy():
 * - Garante que tensores temporários serão descartados automaticamente,
 *   evitando vazamento de memória.
 */
function preprocessImage(input) {
    return tf.tidy(() => {
        const image = tf.browser.fromPixels(input)

        return tf.image
            .resizeBilinear(image, [INPUT_MODEL_DIMENTIONS, INPUT_MODEL_DIMENTIONS])
            .div(255)
            .expandDims(0)
    })
}

async function runInference(tensor) {
    const output = await _model.executeAsync(tensor)
    tf.dispose(tensor)
    // Assume que as 3 primeiras saídas são:
    // caixas (boxes), pontuações (scores) e classes

    const [boxes, scores, classes] = output.slice(0, 3)
    const [boxesData, scoresData, classesData] = await Promise.all(
        [
            boxes.data(),
            scores.data(),
            classes.data(),
        ]
    )

    output.forEach(t => t.dispose())

    return {
        boxes: boxesData,
        scores: scoresData,
        classes: classesData
    }
}

/**
 * Função geradora (function*):
 * Permite retornar valores progressivamente usando `yield`, pausando a execução
 * até que ela seja retomada com `.next()`.
 *
 * Retorna um iterador, permitindo controlar o consumo dos valores e interromper
 * o processamento antes de percorrer todos os elementos.
 *
 * Útil para gerar valores sob demanda, criar iteradores personalizados e
 * processar sequências sem precisar armazenar todos os resultados na memória.
 * 
 * function*: declara uma função geradora.
 * yield: retorna um valor temporariamente e pausa a execução.
 * .next(): retoma a execução até encontrar o próximo yield ou finalizar.
 * done: indica se a execução terminou (true) ou não (false).
 * 
 *  * Filtra e processa as predições:
 * - Aplica o limiar de confiança (CLASS_THRESHOLD)
 * - Filtra apenas a classe desejada (exemplo: 'kite')
 * - Converte coordenadas normalizadas para pixels reais
 * - Calcula o centro do bounding box
 */
function* processPrediction({ boxes, scores, classes }, width, height) {
    for (let i = 0; i < scores.length; i++) {
        if (scores[i] < CLASS_THRESHOLD) continue

        const label = _labels[classes[i]]

        if (label !== 'kite') continue // O pato é considerado como kite porque o modelo foi treinado com o COCO dataset, que não possui pato como classe.

        // Converte as coordenadas da caixa delimitadora de [0, 1] para pixels
        let [x1, y1, x2, y2] = boxes.slice(i * 4, (i + 1) * 4)

        x1 *= width
        y1 *= height
        x2 *= width
        y2 *= height

        const boxWidth = x2 - x1
        const boxHeight = y2 - y1
        const centerX = x1 + boxWidth / 2
        const centerY = y1 + boxHeight / 2

        yield {
            x: centerX,
            y: centerY,
            score: (scores[i] * 100).toFixed(2)
        }
    }
}

loadModelAndLabels()

self.onmessage = async ({ data }) => {
    if (data.type !== 'predict') return
    if (!_model) return

    const input = preprocessImage(data.image)
    const { width, height } = data.image

    const inferenceResults = await runInference(input)

    for (const prediction of processPrediction(inferenceResults, width, height)) {
        postMessage({
            type: 'prediction',
            x: prediction.x,
            y: prediction.y,
            score: prediction.score
        });
    }

};

console.log('🧠 YOLOv5n Web Worker initialized');
