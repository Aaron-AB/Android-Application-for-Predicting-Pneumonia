package com.example.pneumonialite;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    ImageView display;
    Button cameraBtn;
    Button classifyBtn;
    Button quantBtn;
    public static int CAMERA_REQUEST_CODE = 102;

    public static final int imgheight = 224;
    public static final int imgwidth = 224;
    public static final int size = 3;
    private ByteBuffer imgData = ByteBuffer.allocateDirect(imgheight * imgwidth * size * 4);//Multiply by 4 since every Float32 element occupies 4bytes
    private ByteBuffer quantData = ByteBuffer.allocateDirect(imgheight * imgwidth * size * 4);

    public static int IMAGE_MEAN = 128;//Used to normalize inputs
    public static float IMAGE_STD = 128.0f;//Used to normalize inputs

    public int[] intValues = new int[imgheight * imgwidth];//Accounts for all pixels
    public int[] quantValues = new int[imgheight * imgwidth];//Accunt for all pixels - Quant

    private static final String MODEL_PATH  = "PneumoniaLite.tflite";
    private static final String MODEL_PATH2 = "PneumoniaQuant.tflite";
    public static List<String> labels = Arrays.asList("Normal", "Pneumonia");
    private Interpreter tflite;





    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        display = findViewById(R.id.displayImageView);
        classifyBtn = findViewById(R.id.classifyBtn);
        cameraBtn = findViewById(R.id.cameraBtn);
        quantBtn = findViewById(R.id.quantBtn);

        imgData.order(ByteOrder.nativeOrder());
        quantData.order(ByteOrder.nativeOrder());



        cameraBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                OpenCamera();
            }
        });

        classifyBtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Bitmap newimage = ((BitmapDrawable)display.getDrawable()).getBitmap();
                imgData.rewind();
                conversion(newimage);//converts bitmap image into bytebuffer
                float[][] labelProbArray;
                labelProbArray = new float[1][labels.size()];
                final Interpreter.Options tfliteOptions = new Interpreter.Options();
                tfliteOptions.setAllowBufferHandleOutput(true);
                try {
                    tflite = new Interpreter(loadModelFile(MODEL_PATH), tfliteOptions);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                tflite.run(imgData, labelProbArray);

                String val1 = String.format("Normal: %.5f, Pneumonia: %.5f", labelProbArray[0][0], labelProbArray[0][1]);
                Toast.makeText(getApplicationContext(), val1,Toast.LENGTH_LONG).show();
            }


        });

        quantBtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Bitmap newimage = ((BitmapDrawable)display.getDrawable()).getBitmap();
                quantConversion(newimage);
                float[][] quantArray;
                quantArray = new float[1][labels.size()];

                final Interpreter.Options tfliteOptions = new Interpreter.Options();
                tfliteOptions.setAllowBufferHandleOutput(false);

                try {
                    tflite = new Interpreter(loadModelFile(MODEL_PATH2), tfliteOptions);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                tflite.run(quantData, quantArray);
                float value= (quantArray[0][0]);
                float value2= (quantArray[0][1]);
                String val1 = String.format("Normal: %.5f, Pneumonia: %.5f", value, value2);
                Toast.makeText(getApplicationContext(), val1,Toast.LENGTH_LONG).show();
            }
        });

    }


    private void OpenCamera(){
        Intent camera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(camera, CAMERA_REQUEST_CODE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == CAMERA_REQUEST_CODE) {
            Bitmap image = (Bitmap) data.getExtras().get("data");
            Bitmap resizedimg = Bitmap.createScaledBitmap(image, imgheight, imgwidth, false);
            display.setImageBitmap(resizedimg);//sets the background as image
        }

    }

    //Taken from https://blog.tensorflow.org/2018/03/using-tensorflow-lite-on-android.html
    private void conversion(Bitmap bitmap){
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for(int i = 0; i < imgwidth; ++i){
            for(int j = 0; j < imgheight; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                imgData.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                imgData.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
            }
        }
    }

    private void quantConversion(Bitmap bitmap){
        quantData.rewind();
        bitmap.getPixels(quantValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel2 = 0;
        for(int i = 0; i < 224; ++i){
            for(int j = 0; j < 224; ++j) {
                final int val = quantValues[pixel2++];
                quantData.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                quantData.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                quantData.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
            }
        }
    }

    //Taken from https://blog.tensorflow.org/2018/03/using-tensorflow-lite-on-android.html
    private MappedByteBuffer loadModelFile(String path) throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(path);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

    }
}
