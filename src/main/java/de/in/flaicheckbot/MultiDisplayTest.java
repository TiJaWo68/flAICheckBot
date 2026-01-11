package de.in.flaicheckbot;

import java.awt.EventQueue;

import javax.swing.JFrame;

public class MultiDisplayTest {

	public static void main(String[] args) {
		EventQueue.invokeLater(() -> {
			JFrame frame = new JFrame("MultiDisplayTest");
			frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			frame.setBounds(4480, 3, 2150, 1877);
			frame.setVisible(true);
			EventQueue.invokeLater(() -> frame.setBounds(4480, 3, 2150, 1877));
		});
	}

}
