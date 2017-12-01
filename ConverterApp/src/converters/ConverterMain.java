/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package converters;

import java.util.Date;

/**
 *
 * @author olu
 */
public class ConverterMain {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        String baseFolderDir="/home/olu/Dev/data_base/sign_base/training"; //Change as appropriate to you
        String labelingFileDir="/home/olu/Dev/data_base/sign_base/labels.txt"; //Change as appropriate to you
        String formatName = "jpeg";
        
        //Convert Folder Names
        Converter convert = new Converter();
        boolean converted = convert.convertFolderName(baseFolderDir, labelingFileDir);
     
        //Convert all of datasets .ppm to .jpeg
        long timeStart = System.currentTimeMillis();
        boolean converted2 = convert.convertAllDatasetImages(baseFolderDir, formatName);
        if(converted2==true){
        long timeEnd = System.currentTimeMillis(); //in milliseconds
	long diff = timeEnd - timeStart;
	long diffSeconds = diff / 1000 % 60;
	long diffMinutes = diff / (60 * 1000) % 60;
        long diffHours = diff / (60 * 60 * 1000) % 24;
	long diffDays = diff / (24 * 60 * 60 * 1000);
        System.out.println("ALL "+formatName+" CONVERSIONS NOW COMPLETED. Took "+diffDays+" Day(s), "+diffHours+" Hour(s) "+diffMinutes+" Minute(s) and "+diffSeconds+" Second(s)");
        }
        
    }
    
}
