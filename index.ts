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
        const pixelData: tf.PixelData = {
            width: 160,
            height: 250,
            data: new Uint8Array(imageBuffer)
        }
    
        const tensor = tf.browser
            .fromPixels(pixelData)
            .resizeNearestNeighbor([150, 150])
            .expandDims();
        
        const prediction = this.model.predict(tensor);
        const label = (prediction as tf.Tensor).argMax(1).dataSync()[0];

        switch (label) {
            case 1:
                console.log(fileName, 'Cat');
                break;
            case 0:
                console.log(fileName, 'Dog');
                break;
        }
    }

}

const execurtor = new Executor();
execurtor.loadModel().then(() => {
    execurtor.infer();
});

