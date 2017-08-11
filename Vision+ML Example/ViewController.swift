/*
See LICENSE folder for this sample’s licensing information.

Abstract:
View controller for selecting images and applying Vision + Core ML processing.
*/

import UIKit
import CoreML
import Vision
import ImageIO

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var correctedImageView: UIImageView!
    @IBOutlet weak var classificationLabel: UILabel!

    @IBAction func takePicture(_ sender: Any) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .camera
        picker.cameraCaptureMode = .photo
        present(picker, animated: true)
    }
    @IBAction func chooseImage(_ sender: Any) {
        // The photo library is the default source, editing not allowed
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .savedPhotosAlbum
        present(picker, animated: true)
    }

    var inputImage: CIImage! // The image to be processed.

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        picker.dismiss(animated: true)
        classificationLabel.text = "Analyzing Image…"
        correctedImageView.image = nil
        
        guard let uiImage = info[UIImagePickerControllerOriginalImage] as? UIImage
            else { fatalError("no image from image picker") }
        guard let ciImage = CIImage(image: uiImage)
            else { fatalError("can't create CIImage from UIImage") }
        let orientation: CGImagePropertyOrientation = CGImagePropertyOrientation(uiImage.imageOrientation)
        inputImage = ciImage.oriented(forExifOrientation: Int32(orientation.rawValue))

        // Show the image in the UI.
        imageView.image = uiImage

        // Run the rectangle detector, which upon completion runs the ML classifier.
        let handler = VNImageRequestHandler(ciImage:ciImage, orientation: orientation)
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([self.rectanglesRequest])
            } catch {
                print(error)
            }
        }
    }

    lazy var classificationRequest: VNCoreMLRequest = {
        // Load the ML model through its generated class and create a Vision request for it.
        do {
            let model = try VNCoreMLModel(for: OCRClassifier().model)
            return VNCoreMLRequest(model: model, completionHandler: self.handleClassification)
        } catch {
            fatalError("can't load Vision ML model: \(error)")
        }
    }()

    func handleClassification(request: VNRequest, error: Error?) {
        guard let observations = request.results as? [VNClassificationObservation]
            else { fatalError("unexpected result type from VNCoreMLRequest") }
        guard let best = observations.first
            else { fatalError("can't get best result") }

        DispatchQueue.main.async {
            self.classificationLabel.text = "Classification: \"\(best.identifier)\" Confidence: \(best.confidence)"
        }
    }
    
    lazy var rectanglesRequest: VNDetectRectanglesRequest = {
        return VNDetectRectanglesRequest(completionHandler: self.handleRectangles)
    }()
    func handleRectangles(request: VNRequest, error: Error?) {
        guard let observations = request.results as? [VNRectangleObservation]
            else { fatalError("unexpected result type from VNDetectRectanglesRequest") }
//        guard let detectedRectangle = observations.first else {
//            DispatchQueue.main.async {
//                self.classificationLabel.text = "No rectangles detected."
//            }
//            return
//        }
//        let imageSize = inputImage.extent.size

        // Verify detected rectangle is valid.
//        let boundingBox = detectedRectangle.boundingBox.scaled(to: imageSize)
//        guard inputImage.extent.contains(boundingBox)
//            else { print("invalid detected rectangle"); return }

        // Rectify the detected image and reduce it to inverted grayscale for applying model.
//        let topLeft = detectedRectangle.topLeft.scaled(to: imageSize)
//        let topRight = detectedRectangle.topRight.scaled(to: imageSize)
//        let bottomLeft = detectedRectangle.bottomLeft.scaled(to: imageSize)
//        let bottomRight = detectedRectangle.bottomRight.scaled(to: imageSize)
        let correctedImage = inputImage
//            .cropped(to: boundingBox)
//            .applyingFilter("CIPerspectiveCorrection", parameters: [
//                "inputTopLeft": CIVector(cgPoint: topLeft),
//                "inputTopRight": CIVector(cgPoint: topRight),
//                "inputBottomLeft": CIVector(cgPoint: bottomLeft),
//                "inputBottomRight": CIVector(cgPoint: bottomRight)
//            ])
            .applyingFilter("CIColorControls", parameters: [
                kCIInputSaturationKey: 0.5,
                kCIInputContrastKey: 4
            ])
//            .applyingFilter("CIColorInvert")

        // Show the pre-processed image
        DispatchQueue.main.async {
            self.correctedImageView.image = UIImage(ciImage: correctedImage)
        }
        

        var pixelBuffer: CVPixelBuffer? = nil
        CVPixelBufferCreate(kCFAllocatorDefault, Int(correctedImage.extent.size.width), Int(correctedImage.extent.height), kCVPixelFormatType_OneComponent8, nil, &pixelBuffer)
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue:0))

        let imageSide = 50
        let transform = CGAffineTransform(scaleX: CGFloat(imageSide) / CGFloat(correctedImage.extent.width), y: CGFloat(imageSide) / CGFloat(correctedImage.extent.height))
        let scaledImage = correctedImage.transformed(by: transform).cropped(to: CGRect(x: 0, y: 0, width: imageSide, height: imageSide))
        let mlMultiArray = extractFeatures(fromCGImage: convertCIImageToCGImage(inputImage: scaledImage))
        
        let model = OCRClassifier()
     
        guard let predictionOutput = try? model.prediction(imageArray: mlMultiArray!) else {
            fatalError("Unexpected runtime error. model.prediction")
        }
        print(predictionOutput.character)
        DispatchQueue.main.async {
            self.classificationLabel.text = "Classification: \"\(predictionOutput.character)\" Confidence: \(predictionOutput.classProbability[predictionOutput.character])"
        }
        
    }
    
    /// resize CVPixelBuffer
    ///
    /// - Parameter pixelBuffer: CVPixelBuffer by camera output
    /// - Returns: CVPixelBuffer with size (299, 299)
    func resize(pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        let imageSide = 50
        var ciImage = CIImage(cvPixelBuffer: pixelBuffer, options: nil)
        let transform = CGAffineTransform(scaleX: CGFloat(imageSide) / CGFloat(CVPixelBufferGetWidth(pixelBuffer)), y: CGFloat(imageSide) / CGFloat(CVPixelBufferGetHeight(pixelBuffer)))
        ciImage = ciImage.transformed(by: transform).cropped(to: CGRect(x: 0, y: 0, width: imageSide, height: imageSide))
        let ciContext = CIContext()
        var resizeBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, imageSide, imageSide, CVPixelBufferGetPixelFormatType(pixelBuffer), nil, &resizeBuffer)
        ciContext.render(ciImage, to: resizeBuffer!)
        return resizeBuffer
    }
    
    func convertCIImageToCGImage(inputImage: CIImage) -> CGImage! {
        let context = CIContext(options: nil)
        
        return context.createCGImage(inputImage, from: inputImage.extent)
    }
    
    // Converts an image to an array of pixel values.
    // Trims the padding generated by the image stride.
    // Converts the image to grayscale and normalized the values
    func extractFeatures(fromCGImage imageRef: CGImage?) -> MLMultiArray?
    {
        var width = 0
        var height = 0
        var pixelValues: MLMultiArray?
        if let imageRef = imageRef {
            width = imageRef.width
            height = imageRef.height
            let bitsPerComponent = imageRef.bitsPerComponent
            let bytesPerRow = imageRef.bytesPerRow
            let totalBytes = height * bytesPerRow
            
            let colorSpace = CGColorSpaceCreateDeviceGray()
            var intensities = [UInt8](repeating: 0, count: totalBytes)
            
            let contextRef = CGContext(data: &intensities, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: 0)
            contextRef?.draw(imageRef, in: CGRect(x: 0.0, y: 0.0, width: CGFloat(width), height: CGFloat(height)))
            
            guard let mlMultiArray = try? MLMultiArray(shape:[2500], dataType:MLMultiArrayDataType.double) else {
                fatalError("Unexpected runtime error. MLMultiArray")
            }
            var correctedIndex = 0
            for (index, element) in intensities.enumerated() {
                if index%bytesPerRow < 50 {
                    mlMultiArray[correctedIndex] = NSNumber(floatLiteral: Double(element))
                    correctedIndex += 1
                }
            }
            
            pixelValues = mlMultiArray
        }
        return pixelValues
    }

}

