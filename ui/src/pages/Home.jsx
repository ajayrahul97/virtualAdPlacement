import React from 'react';

import Header from '../partials/Header';
import PageIllustration from '../partials/PageIllustration';
import HeroHome from '../partials/HeroHome';
import FeaturesBlocks from '../partials/FeaturesBlocks';
import FeaturesZigZag from '../partials/FeaturesZigzag';
import FeaturesZigZagArch from '../partials/FeaturesZigzagArch';
import FeaturesZigZagMore from '../partials/FeaturesZigZagMore';
import FeaturesZigZagOriginal from '../partials/FeaturesZigZagOriginal';
import FeaturesZigZagLocal from '../partials/FeaturesZigZagLocal';
import FeaturesZigZagNational from '../partials/FeaturesZigZagNational';
import FeaturesZigZagPod1 from '../partials/FeaturesZigZagPod1';
import FeaturesZigZagNationalPod2 from '../partials/FeaturesZigZagNationalPod2';
import FeaturesZigZagNationalPod3 from '../partials/FeaturesZigZagNationalPod3';

import Testimonials from '../partials/Testimonials';
import Newsletter from '../partials/Newsletter';
import Banner from '../partials/Banner';
import Footer from '../partials/Footer';

function Home() {
  return (
    <div className="flex flex-col min-h-screen overflow-hidden">
      {/*  Site header */}
      <Header />

      {/*  Page content */}
      <main className="grow">
        {/*  Page illustration */}
        <div className="relative max-w-6xl mx-auto h-0 pointer-events-none" aria-hidden="true">
          <PageIllustration />
        </div>

        {/*  Page sections */}
        <HeroHome />
        <FeaturesBlocks />
        <FeaturesZigZagOriginal />
        <FeaturesZigZagLocal/>
        <FeaturesZigZagNational/>
        <FeaturesZigZagPod1/>
        <FeaturesZigZagNationalPod2/>
        <FeaturesZigZagNationalPod3/>
        <FeaturesZigZagArch/>        
        <Testimonials />
        <FeaturesZigZagMore/>

      </main>

      <Banner />

      {/*  Site footer */}
      <Footer />
    </div>
  );
}

export default Home;