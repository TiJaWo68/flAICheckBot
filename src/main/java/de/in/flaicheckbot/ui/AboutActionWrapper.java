package de.in.flaicheckbot.ui;

import java.awt.Image;
import java.awt.event.ActionEvent;

import javax.swing.JFrame;

import org.jfree.base.Library;
import org.jfree.ui.about.ProjectInfo;

import de.in.utils.Version;
import de.in.utils.gui.AboutAction;
import de.in.utils.gui.AboutDialog;

/**
 * Enhanced AboutAction that includes backend Open Source libraries.
 */
public class AboutActionWrapper extends AboutAction {

        public AboutActionWrapper(JFrame parent, String copyrightHolder, String url, String projectName, int startYear,
                        String groupId, String artifactId, Image image) {
                super(parent, copyrightHolder, url, projectName, startYear, groupId, artifactId, image);
        }

        @Override
        public void actionPerformed(ActionEvent e) {
                ProjectInfo project = Version.retrieveProjectInfoFromPom(copyrightHolder, startYear, groupId,
                                artifactId);
                project.setName(projectName);
                project.setLogo(image);

                // Add Backend Libraries
                project.addLibrary(new Library("FastAPI", "0.115.6", "FastAPI", "https://fastapi.tiangolo.com/"));
                project.addLibrary(new Library("Uvicorn", "0.34.0", "Uvicorn", "https://www.uvicorn.org/"));
                project.addLibrary(new Library("Pillow", "11.0.0", "Pillow", "https://python-pillow.org/"));
                project.addLibrary(new Library("PyTorch", "2.5.1", "PyTorch", "https://pytorch.org/"));
                project.addLibrary(
                                new Library("Transformers", "4.47.1", "Transformers",
                                                "https://huggingface.co/docs/transformers/"));
                project.addLibrary(new Library("OpenCV-Python", "4.10.0.84", "OpenCV-Python", "https://opencv.org/"));
                project.addLibrary(new Library("Google Cloud AI Platform", "1.34.1", "Google Cloud AI Platform",
                                "https://github.com/googleapis/python-aiplatform"));
                project.addLibrary(new Library("Google Generative AI", "0.8.3", "Google Generative AI",
                                "https://github.com/google/generative-ai-python"));

                AboutDialog dialog = new AboutDialog(parent, "", project, url);
                dialog.pack();
                dialog.setLocationRelativeTo(parent);
                dialog.setVisible(true);
        }
}
