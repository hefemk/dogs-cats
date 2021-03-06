import * as tf from '@tensorflow/tfjs';
import * as tfn from '@tensorflow/tfjs-node';
import * as fs from 'fs';

class Executor {

    private model!: tf.LayersModel;

    async loadModel(): Promise<void> {
        const handler = tfn.io.fileSystem("./models/model.json");
        this.model = await tf.loadLayersModel(handler);
        this.model.summary();
    }

    async infer(): Promise<void> {
        const filesInDir: string[] = fs.readdirSync('./images');
        filesInDir
        .filter((file: string) => {
            return file.endsWith('.jpg');
        })
        .forEach(async (file: string) => {
            const path: string = `./images/${file}`;
            const buffer = fs.readFileSync(path);
            await this.inferImage(buffer, file);
        });
    }

    private async inferImage(imageBuffer: Buffer, fileName: string): Promise<void> {
        const imgDataArray = new Uint8Array(imageBuffer);

        let tensor = tfn.node
            .decodeJpeg(imgDataArray)
            .resizeNearestNeighbor([224, 224]);
        // await this.saveProcessedImage(tensor, fileName);

        tensor = tensor
            .sub(127)
            .div(127)
            .expandDims();
        const prediction = this.model.predict(tensor);
        
        const label = (prediction as tf.Tensor).argMax(1).dataSync()[0];

        switch (label) {
            case 0:
                console.log(fileName, 'Cat');
                break;
            case 1:
                console.log(fileName, 'Dog');
                break;
        }
    }

    private async saveProcessedImage(tensor: tf.Tensor, imageName: string): Promise<void> {
        const encoded = await tfn.node.encodeJpeg(tensor as any);
        fs.writeFileSync(`./images/processed/${imageName}`, encoded);
    }

}

const execurtor = new Executor();
execurtor.loadModel().then(() => {
    execurtor.infer();
});

