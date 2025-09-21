import React from 'react';
import './Header.css';

interface HeaderProps {
  onStatusClick: () => void;
  onAboutClick: () => void;
}

const Header: React.FC<HeaderProps> = ({ onStatusClick, onAboutClick }) => {
  return (
    <header>
      <h1 onClick={onAboutClick}>train🚂...or no train?</h1>
      <nav>
        <a href="#status" onClick={onStatusClick}>status</a>
        <a href="#about" onClick={onAboutClick}>about</a>
      </nav>
    </header>
  );
};

export default Header;
