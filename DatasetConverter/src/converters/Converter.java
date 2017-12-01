/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package converters;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.Scanner;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

/**
 *
 * @author olu
 * email: oluwoleoyetoke@gmail.com
 * Class contains methods used to 
 * 1. Convert GTSB dataset folder names to labels for the class
 * 2. Convert the datasets .ppm images to .jpeg
 * (GTSB - German Traffic Sign Benchmark)
 */
public class Converter implements Runnable {
    String pathToFolderUnderView=null;
    String format;
    static int totalConversion;
    
    //Default constructor
    Converter(){
        
    }
    
    //Constructor used when multithread operation is needed for the picture conversion
    Converter(String pathToFolderUnderView, String format){
        this.pathToFolderUnderView= pathToFolderUnderView;
        this.format = format;
    }
    
    @Override
    public void run(){
        System.out.println("(Multithreaded) Now Converting Contents of Folder "+this.pathToFolderUnderView);
        //Connect to sub folder and begin to convert all its subfiles
        long timeStart = System.currentTimeMillis();
		
        File file = new File(this.pathToFolderUnderView);
        String extension="";
        String[] fileNames = file.list();
        for(int i=0; i<fileNames.length; i++){
            extension = getFileExtensionFromPath(this.pathToFolderUnderView+"/"+fileNames[i]);
            if(!extension.equals("ppm")){
                continue;
            }
            //Do Conversion
            try{
            convertImageFormat2(this.pathToFolderUnderView+"/"+fileNames[i], format);
            totalConversion++;
            }catch(Exception ex){
                System.out.println("Encountered Error While Converting: "+ex.getMessage());
                System.exit(1);
            }
            
            //Delete previous format
            File toDelete = new File(this.pathToFolderUnderView+"/"+fileNames[i]);
            toDelete.delete();
            
        }
        
        long timeEnd = System.currentTimeMillis(); //in milliseconds
	long diff = timeEnd - timeStart;
	long diffSeconds = diff / 1000 % 60;
	long diffMinutes = diff / (60 * 1000) % 60;
        long diffHours = diff / (60 * 60 * 1000) % 24;
	long diffDays = diff / (24 * 60 * 60 * 1000);
        System.out.println("Done Converting the Content of Folder: "+this.pathToFolderUnderView+". Took "+diffDays+" Day(s), "+diffHours+" Hour(s) "+diffMinutes+" Minute(s) and "+diffSeconds+" Second(s)");
           
    }
    
    //Convert all the datasets image to another format e.g .png
    public boolean convertAllDatasetImages(String baseFolderDir, String format){
        
        //Validation
        if(baseFolderDir.isEmpty()){
            System.out.println("baseFolderDir not set");
            return false;
        }else if(format.isEmpty()){
            System.out.println("Please specify a format");
            return false;
        }else if(!getFormats().contains(format)){
            System.out.println("Please specify a compatible format to convert to e.g "+getFormats().toString());
            return false;
        }
        
        //User advice
        System.out.println("It is adviced that you have only folders in the "
                + "base directory containing your training images.\n"
                + "Base folder-->Subfolders-->Each subfolder containing "
                + "specific classes of image\n"
                + "E.g Training Folder -> stop_sign_folder -> 1.jpg, 2.jpg, 3.jpg....");
        
        //Get user confirmation
        Scanner scanner = new Scanner(System.in);
        System.out.println("Would you like to proceed? [Y/N]: ");
        String answer = scanner.next();
        if(!answer.equals("Y")){
            System.out.println("Exiting....");
            return false;
        }
        
        //Connect to dataset base deirectory
        File base =null;
        try{ //For saftey sake
           base = new File(baseFolderDir);
         if(!base.isDirectory()){ //Check to make sure directory specified isnt just a file
            System.out.println("Not a directory");
            return false;
        }
        }catch (Exception ex){
            System.out.println("Error occured while opening directory: "+ex.getMessage());
            return false;
        }
        
        System.out.println("Base Directory: "+baseFolderDir);
        
        //Get sub directories
         String[] subFiles =  base.list();
         File[] subFilesHandle = base.listFiles();  
         
      
        //Confirm that base directory has sub directories or at least, sub files
        int noOfContents = 0;
        noOfContents = subFiles.length;
        System.out.println("Number of sub directories or posibly files: "+noOfContents);
        if(noOfContents==0){
             System.out.println("There are no sub files/directories in the base directory");
            return false;
        }
        
        System.out.println("About to begin multithreaded conversion. Please note, this might take some time");
        String pathToFolderUnderOperation ="";
        //Open each subdirectory and convert image present to desired format (Multi Threaded)
        //Use executors to manage concurrency
        ExecutorService executor = Executors.newCachedThreadPool();
        for(int i=0; i<noOfContents; i++){
            File folderUnderView = new File(baseFolderDir+"/"+subFiles[i]);
            if(!folderUnderView.isDirectory()){ //confirm that it is a directory
                continue; //Move to the next itteration
            }
            pathToFolderUnderOperation = folderUnderView.getPath();
            Converter conv = new Converter(pathToFolderUnderOperation, format);
            //Submit task to executor
            executor.submit(conv);
            //Thread thread = new Thread(conv);
            //thread.start();   
        }
        
        executor.shutdown(); //Shutdown executor when done
        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.HOURS); //wait for 1 hour * long max
        } catch (InterruptedException e) {
            System.out.println("Error while awaiting executor shutdown: "+e.getMessage());
        }
        return true;       
    }
    
    //Convert all the datasets folder names to class labels...Needed for TensorFlow
    public boolean convertFolderName(String baseFolderDir, String labelingFileDir){
	File dir = null; 
        
        //User advice
        System.out.println("It is adviced that you have only folders in the "
                + "base directory containing your training images.\n"
                + "Base folder-->Subfolders-->Each subfolder containing "
                + "specific classes of image\n"
                + "E.g Training Folder -> stop_sign_folder -> 1.jpg, 2.jpg, 3.jpg....");
        
        //Get user confirmation
        Scanner scanner = new Scanner(System.in);
        System.out.println("Would you like to proceed? [Y/N]: ");
        String answer = scanner.next();
        if(!answer.equals("Y")){
            System.out.println("Exiting....");
            return false;
        }
        
        //Validation
        if(baseFolderDir.isEmpty()){
            System.out.println("baseFolderDir not set");
            return false;
        }else if(labelingFileDir.isEmpty()){
            System.out.println("blabelingFileDir not set");
            return false;
        }
        
        //Try to open directory
        try{ //For saftey sake
        dir = new File(baseFolderDir);
         if(!dir.isDirectory()){ //Check to make sure directory specified isnt just a file
            System.out.println("Not a directory");
            return false;
        }
        }catch (Exception ex){
            System.out.println("Error occured while opening directory: "+ex.getMessage());
            return false;
        }
         
         //Get sub directories
         String[] subFiles =  dir.list();
         File[] subFilesHandle = dir.listFiles();         
         
         // Sort files/folders handle by name
         Arrays.sort(subFilesHandle, new Comparator(){
             @Override
                     public int compare(Object f1, Object f2){
                         return ((File) f1).getName().compareTo(((File) f2).getName());
                     }
         });
         Arrays.sort(subFiles); //sort files/folders string by name
         ArrayList subDirs = new ArrayList();
         File test = null;
         int noOfContents = subFiles.length;
         for (int i=0; i<noOfContents; i++){
             test = new File(baseFolderDir+"/"+subFiles[i]);
             if(test.isDirectory()){
                 subDirs.add(subFiles[i]);
             }else{
                 System.out.println("One or more of the subfiles ("+test.getName()+") is not a directory");
                 return false;
             }
         }
         int noOfSubFlders = subDirs.size();
         
         //Load labeling txt
         File labelFile;
         BufferedReader buff;
         try{//Check to make sure labels file is an actual file
         labelFile = new File (labelingFileDir);
         if(labelFile.isFile()==false){
             System.out.println("Not a file");
             return false;
         }
         }catch(Exception ex){
            System.out.println("Error Encountered while opening labeling file: "+ex.getMessage());
            return false;
         }
         
         //Reading Labels
         ArrayList labels = new ArrayList();
         BufferedReader buffRead = null;
         try{
             String line;
             buffRead = new BufferedReader(new FileReader(labelingFileDir));
             while((line=buffRead.readLine())!=null){
                 labels.add(String.valueOf(line));
             }
         }catch(Exception ex){
             System.out.println("Error while reading labels: "+ex.getMessage());
             return false;
         }
         int noOfLabels = labels.size();
         
         if(noOfLabels!=noOfSubFlders){
             System.out.println("Number of labels and subfolders not equal");
             return false; 
         }
         
         //Check to be sure selections are same with labels position, and as user wants it
         String name="";
         File newName;
         for(int i=0; i<noOfSubFlders; i++){
             name = subFilesHandle[i].getName();
             if( subDirs.contains(name)){
                 newName = new File(labels.get(i).toString());
                 //subFilesHandle[i].renameTo(newName);
                 System.out.println("Folder "+name+" will be renamed to "+newName.getName());
             } 
         }
         
         //Get user final confirmation
        System.out.println("Is this renaming what you want? [Y/N]: ");
        answer = scanner.next();
        if(!answer.equals("Y")){
            System.out.println("Exiting....");
            return false;
        }
         
         //relabeling the folders
         String path="";
         for(int i=0; i<noOfContents; i++){
             name = subFilesHandle[i].getName();
             if( subDirs.contains(name)){
                 path = subFilesHandle[i].getParent()+"/"+labels.get(i).toString();
                 //System.out.println(path);
                 newName = new File(path);
                 subFilesHandle[i].renameTo(newName);
                 System.out.println("Folder "+name+" has been renamed to "+newName.getName());
             }   
         }
         System.out.println("Relabeling completed"); 
         return true;
    }
    
    
    //Normal Java Image Converter....Does not support .ppm conversion
    public boolean convertImageFormat(String inputImagePath, String outputImagePath, String formatName) throws IOException {
        FileInputStream inputStream = new FileInputStream(inputImagePath);
        FileOutputStream outputStream = new FileOutputStream(outputImagePath);
         
        // reads input image from file
        BufferedImage inputImage = ImageIO.read(inputStream);
         
        // writes to the output image in specified format
        boolean result = ImageIO.write(inputImage, formatName, outputStream);
         
        // needs to close the streams
        outputStream.close();
        inputStream.close();
         
        return result;
    }
    
    
    //Image converter gotten from: https://github.com/Jpadilla1/Steganography/blob/master/src/imageconverter/ImageConverter.java
     public static String convertImageFormat2(String inputFilePath, String format) throws Exception {
        if (getFileExtensionFromPath(inputFilePath).equals(format)) {
            throw new Exception("Formats can't be the same!");
        }
        if (getFormats().contains(format)
                && getFormats().contains(getFileExtensionFromPath(inputFilePath))) {

            String outputPath = getOutputPathFromInputPath(inputFilePath, format);
            String convertPath = "";
            if (System.getProperty("os.name").equals("Linux"))
            {
                convertPath = "/usr/bin/convert";
            } else {
                convertPath = "/usr/local/bin/convert";
            }

            if (format.equals("ppm")) {
                executeCommand(convertPath + " -compress none " + inputFilePath + " " + getOutputPathFromInputPath(inputFilePath, format));
            } else {
                executeCommand(convertPath + " " + inputFilePath + " " + getOutputPathFromInputPath(inputFilePath, format));
            }
            
            return outputPath;
        } else {
            throw new Exception("Format not yet implemented");
        }
    }

    private static ArrayList<String> getFormats() {
        return new ArrayList<>(Arrays.asList("jpg", "jpeg", "png", "ppm", "gif"));
    }

    private static String getFileExtensionFromPath(String path) {
        int i = path.lastIndexOf('.');
        if (i > 0) {
            return path.substring(i + 1);
        }
        return "";
    }

    private static String getOutputPathFromInputPath(String path, String format) {
        return path.substring(0, path.lastIndexOf('.')) + "." + format;
    }

    private static String executeCommand(String command) {

        StringBuilder output = new StringBuilder();

        Process p;
        try {
            p = Runtime.getRuntime().exec(command);
            p.waitFor();
            BufferedReader reader
                    = new BufferedReader(new InputStreamReader(p.getInputStream()));

            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

        } catch (IOException | InterruptedException ex) {
            Logger.getLogger(Converter.class.getName()).log(Level.SEVERE, null, ex);
        }

        return output.toString();
}
    
    
}
