import React,  { useState, useRef, useEffect } from 'react';
import Modal from '../utils/Modal';

import HeroImage from '../images/kg.jpg';

function FeaturesZigzagOriginal() {
  const [videoModalOpen, setVideoModalOpen] = useState(false);
  const [videoModalOpen1, setVideoModalOpen1] = useState(false);

  const [selectedFile, setSelectedFile] = useState(null);

  const video = useRef(null);
  var videoName = "/videos/kg_original.mp4";
  var localVid = "/videos/kg_local1.mp4";
  var sponsorVid = "/videos/kg_sponsor.mp4";

  useEffect(() => {
    videoModalOpen ? video.current.play() : video.current.pause();
  }, [videoModalOpen]);   

  
  return (
    <section>
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="py-12 md:py-20 border-t border-gray-800">

          {/* Section header */}
          <div className="max-w-3xl mx-auto text-center pb-12 md:pb-16">
            <div className="inline-flex text-sm font-semibold py-1 px-3 m-2 text-green-600 bg-green-200 rounded-full mb-4">Reach goals that matter</div>
            <h1 className="h2 mb-4">One product, unlimited solutions</h1>
          </div>

          {/* Items */}
          <div className="grid gap-20">

            {/* 1st item */}
            <div className="md:grid md:grid-cols-12 md:gap-6 items-center">
              {/* Image */}
              
              <div className="max-w-xl md:max-w-none md:w-full mx-auto md:col-span-5 lg:col-span-6 mb-8 md:mb-0 md:order-1" data-aos="fade-up">
              <div>
            <div className="relative flex justify-center items-center" data-aos="fade-up" data-aos-delay="200">
              <img className="mx-auto" src={HeroImage} width="1024" height="504" alt="Hero" />
              <a
                className="absolute group"
                href="#0"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setVideoModalOpen(true);
                }}
                aria-controls="modal"
              >
                <svg
                  className="w-16 h-16 sm:w-20 sm:h-20 hover:opacity-75 transition duration-150 ease-in-out"
                  viewBox="0 0 88 88"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <defs>
                    <linearGradient x1="78.169%" y1="9.507%" x2="24.434%" y2="90.469%" id="a">
                      <stop stopColor="#EBF1F5" stopOpacity=".8" offset="0%" />
                      <stop stopColor="#EBF1F5" offset="100%" />
                    </linearGradient>
                  </defs>
                  <circle fill="url(#a)" cx="44" cy="44" r="44" />
                  <path
                    className="fill-current text-purple-600"
                    d="M52 44a.999.999 0 00-.427-.82l-10-7A1 1 0 0040 37V51a.999.999 0 001.573.82l10-7A.995.995 0 0052 44V44c0 .001 0 .001 0 0z"
                  />
                </svg>
              </a>
            </div>

            {/* Modal */}
            <Modal id="modal" ariaLabel="modal-headline" show={videoModalOpen} handleClose={() => setVideoModalOpen(false)}>
              <div className="relative pb-9/16">
                <video ref={video} className="absolute w-full h-full" width="1920" height="1080" loop autoPlay controls>
                  <source src={videoName} type="video/mp4" />
                  Your browser does not support the video tag.
                </video>
              </div>
            </Modal>
          </div>
              </div>
              {/* Content */}
              <div className="max-w-xl md:max-w-none md:w-full mx-auto md:col-span-7 lg:col-span-6" data-aos="fade-right">
                <div className="md:pr-4 lg:pr-12 xl:pr-16">
                  <h3 className="h3 mb-3">Original Video</h3>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
}



export default FeaturesZigzagOriginal;