import { useState, useEffect } from 'react';
import { FaCheckCircle, FaTimesCircle } from 'react-icons/fa';
import Header from './components/Header';
import Modal from './components/Modal';
import './App.css';

function App() {
  const [showStatus, setShowStatus] = useState(false);
  const [showAbout, setShowAbout] = useState(false);
  const [lastUpdated, setLastUpdated] = useState('');
  const [isTrain, setIsTrain] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/getTrainStatus')
      .then((res) => res.json())
      .then((data) => {
        setLastUpdated(data.timestamp);
        setIsTrain(data.train);
        setLoading(false);
      });
  }, []);

  return (
    <>
      <Header 
        onStatusClick={() => setShowStatus(true)}
        onAboutClick={() => setShowAbout(true)}
      />
      <main>
        {loading ? (
          <p>Loading...</p>
        ) : (
          <>
            <div className="image-container">
              <img src="https://msft2025trainornotrain.blob.core.windows.net/web/lee-allene-intersection.png" alt="Lee & Allene Intersection" />
              <div className="overlay" onClick={() => setShowStatus(true)}>
                <div className={`overlay-icon ${isTrain ? 'train-icon' : 'check-icon'}`}>
                  <div className="overlay-text">
                    {isTrain ? 'TRAIN' : 'NO TRAIN'}
                  </div>
                  <div className="overlay-symbol">
                    {isTrain ? (
                      <FaTimesCircle />
                    ) : (
                      <FaCheckCircle />
                    )}
                  </div>
                </div>
              </div>
            </div>
            <p>last updated: {new Date(lastUpdated).toLocaleString()}</p>
          </>
        )}
      </main>
      <Modal show={showStatus} onClose={() => setShowStatus(false)} title="Status: Allene Avenue Southwest & Lee Street Southwest- 30310">
        <p>The train is {isTrain ? 'present' : 'not present'}.</p>
      </Modal>
      <Modal show={showAbout} onClose={() => setShowAbout(false)} title={<><i>Fast Track to Clear Tracks</i></>}>
        <p>Fast Track to Clear Tracks is a smart infrastructure solution built to address Atlanta’s ongoing and hazardous problem of freight trains blocking road crossings for extended periods. These idle trains pose serious safety risks by delaying emergency responders and disrupting daily life for residents—especially seniors and those with limited mobility. Leveraging real-time AI-powered image recognition from edge devices and Azure-based cloud automation, the system continuously monitors train activity, alerts railroad operators when crossings are obstructed, and keeps the public informed via social media updates and a dynamic web-based heads-up display.</p>
        <p>Developed by <a href="https://github.com/pocketcalculator" target="_blank" rel="noopener noreferrer">Paul Sczurek</a> for the 2025 Microsoft Hackathon.</p>
      </Modal>
    </>
  );
}

export default App;

