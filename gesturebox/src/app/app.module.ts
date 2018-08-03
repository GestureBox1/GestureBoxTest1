import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { HttpClientModule, HTTP_INTERCEPTORS } from '@angular/common/http';
import { NgModule, APP_INITIALIZER } from '@angular/core';

import { AppLocationsService } from '@kognifai/poseidon-ng-applocationsservice';
import { AuthenticationInterceptor } from '@kognifai/poseidon-ng-authenticationinterceptor';
import { AuthenticationService as AuthenticationSvc } from '@kognifai/poseidon-authenticationservice';
import { AuthenticationService } from '@kognifai/poseidon-ng-authenticationservice';
import { ConfigurationService } from '@kognifai/poseidon-ng-configurationservice';
import { CookieService } from '@kognifai/poseidon-cookieservice';
import { FooterModule } from '@kognifai/poseidon-ng-footer-component';
import { HeaderModule } from '@kognifai/poseidon-header-component';
import { IConfiguration } from '@kognifai/poseidon-configurationinterface';
import { InitializeService } from '@kognifai/poseidon-ng-initialize-service';
import { MessageModule } from '@kognifai/poseidon-ng-message-component';
import { MessageService } from '@kognifai/poseidon-message-service';
import { NavigationService } from '@kognifai/poseidon-ng-navigationservice';
import { NavigationSidebarModule } from '@kognifai/poseidon-navigationsidebar-component';
import { SettingsService } from '@kognifai/poseidon-ng-settingsservice';
import { SidebarsVisibilityService } from '@kognifai/poseidon-sidebar-visibilityservice';
import { ToolsMenuModule } from '@kognifai/poseidon-toolsmenu';

import { AppComponent } from './app.component';
import { AppRoutingModule } from './app-routing.module';
import { AuthenticationGuardService } from './authentication-guard.service';
import { MainComponent } from './main/main.component';
import { HomeComponent } from './home/home.component';

export function initConfig(config: ConfigurationService<IConfiguration>) {
  return () => config.load();
}

@NgModule({
  imports: [
    AppRoutingModule,
    BrowserAnimationsModule,
    BrowserModule,
    FooterModule,
    FormsModule,
    HeaderModule,
    HttpClientModule,
    MessageModule,
    NavigationSidebarModule,
    ReactiveFormsModule,
    ToolsMenuModule,
  ],
  declarations: [
    AppComponent,
    MainComponent,
    HomeComponent,
  ],
  entryComponents: [ ],
  providers: [
    {
        provide: HTTP_INTERCEPTORS,
        useClass: AuthenticationInterceptor,
        multi: true
    },
    {
        provide: APP_INITIALIZER,
        useFactory: initConfig,
        deps: [ConfigurationService],
        multi: true
    },
    AppLocationsService,
    AuthenticationGuardService,
    AuthenticationSvc,
    AuthenticationService,
    ConfigurationService,
    CookieService,
    InitializeService,
    MessageService,
    NavigationService,
    SettingsService,
    SidebarsVisibilityService,
  ],
  bootstrap: [ AppComponent ]
})
export class AppModule { }
