const container = document.querySelector("#container")
const input = document.querySelector("#image-input")

async function loadTrainingData() {
    const labels = ["Antony Starr", "Dwayne Johnson", "Robert Downey Jr", "Tom Cruise"]
    const faceDescriptors = []
    for (const label of labels) {
        const descriptors = []
        for (let i = 1; i<=4; i++){
            const image = await faceapi.fetchImage(`/data/${label}/${i}.jpeg`)
            const detection = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
            descriptors.push(detection[0].descriptor)
        }
        faceDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptors))
    }
    Toastify({text: "Đã training xong dữ liệu các label"}).showToast()
    return faceDescriptors
}

let faceMatcher

async function init() {
    await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
        faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
        faceapi.nets.faceRecognitionNet.loadFromUri("/models")
    ])

    const trainingData = await loadTrainingData()
    faceMatcher = new faceapi.FaceMatcher(trainingData, 0.6)

    Toastify({text: "Đã tải xong các model"}).showToast()
}

init()

input.addEventListener("change", async (event) => {
    const file = input.files[0]
    const image = await faceapi.bufferToImage(file)
    const canvas =  faceapi.createCanvasFromMedia(image)
    container.innerHTML = ""
    container.append(image)
    container.append(canvas)
    const size = {  width : image.width, height: image.height   }
    faceapi.matchDimensions(canvas, size)

    const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
    const resizeDetections = faceapi.resizeResults(detections,size)
    
    for (const detection of resizeDetections) {
        const box = detection.detection.box
        console.log(detection);
        const drawBox = new faceapi.draw.DrawBox(box, {
            label: faceMatcher.findBestMatch(detection.descriptor)
        })
        drawBox.draw(canvas)
    }
})